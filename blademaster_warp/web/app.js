import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
const $ = id => document.getElementById(id);
// Pieces below 1% of the particles count as debris, not as a separated piece.
function showPieces(sizes){const out=$('pieces');if(!sizes.length){out.textContent='—';out.className='';return}
 const total=sizes.reduce((a,b)=>a+b,0),big=sizes.filter(s=>s>=Math.max(5,0.01*total)),debris=sizes.length-big.length;
 out.textContent=big.length+' 块'+(big.length===1?'（未切断）':'')+(debris?` · 碎粒 ${debris}`:'');
 out.className=big.length>1?'good':'';out.title='各块粒子数：'+sizes.join(' / ')}
const renderer = new THREE.WebGLRenderer({canvas:$('canvas'),antialias:true,alpha:false});
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.setClearColor(0xf2f5f1); renderer.setScissorTest(true);
const scene = new THREE.Scene();
scene.add(new THREE.HemisphereLight(0xffffff,0x77806c,2.4));
const light=new THREE.DirectionalLight(0xffffff,2.3);light.position.set(1,-2,3);scene.add(light);
const fill=new THREE.DirectionalLight(0xe7efff,.7);fill.position.set(-1,2,1);scene.add(fill);
for(const l of scene.children)l.layers.enableAll();
const views=['whole','detail','particles'].map((id,i)=>{
 const camera=new THREE.PerspectiveCamera(i?35:38,1,.001,20);camera.up.set(0,0,1);
 if(i===2)camera.layers.set(1);
 const controls=new OrbitControls(camera,$(id));controls.enableDamping=true;
 return {element:$(id),camera,controls};
});
let objects=[],surface=null,particles=null,assetGeneration=-1,seq=-1,lastReceive=0,packetCount=0,lastState='loading';
const timing={fetchMs:[],parseMs:[],applyMs:[],bytes:[],presentMs:[]};let appliedAt=null;
function keep(list,value){list.push(value);if(list.length>200)list.shift()}
const matrix=new THREE.Matrix4(),dummy=new THREE.Object3D();
function installAsset(asset){
 for(const o of objects){scene.remove(o);o.geometry.dispose();o.material.dispose()}
 objects=[];
 for(const g of asset.geoms){
  let geometry;
  if(g.mesh!==undefined){const m=asset.meshes[g.mesh];geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(m.vertices,3));geometry.setAttribute('normal',new THREE.Float32BufferAttribute(m.normals,3))}
  else if(g.kind===6)geometry=new THREE.BoxGeometry(...g.size.map(x=>2*x));
  else if(g.kind===5){geometry=new THREE.CylinderGeometry(g.size[0],g.size[0],g.size[1]*2,48);geometry.rotateX(Math.PI/2)}
  else if(g.kind===2)geometry=new THREE.SphereGeometry(g.size[0],24,16);
  else if(g.kind===3){geometry=new THREE.CapsuleGeometry(g.size[0],g.size[1]*2,8,24);geometry.rotateX(Math.PI/2)}
  else if(g.kind===4){geometry=new THREE.SphereGeometry(1,24,16);geometry.scale(...g.size)}
  else throw Error('Unsupported visual type '+g.kind);
  const material=new THREE.MeshStandardMaterial({color:new THREE.Color(...g.rgba.slice(0,3)),roughness:.6,metalness:.08});
  const mesh=new THREE.Mesh(geometry,material);mesh.matrixAutoUpdate=false;if(['knife','board','knife_handle'].includes(g.name))mesh.layers.enable(1);scene.add(mesh);objects.push(mesh);
 }
 if(surface){scene.remove(surface);surface.geometry.dispose();surface.material.dispose();surface=null}
 if(particles){scene.remove(particles);particles.geometry.dispose();particles.material.dispose();particles=null}
 const origin=new THREE.Vector3(...asset.origin);
 views.forEach((v,i)=>{v.controls.target.copy(i?origin.clone().add(new THREE.Vector3(0,0,.025)):new THREE.Vector3(.25,0,.38));v.camera.position.copy(v.controls.target).add(i?new THREE.Vector3(.18,-.21,.17):new THREE.Vector3(1,-1.3,.85));v.controls.update()});
 assetGeneration=asset.generation;
}
function applyPacket(p){
 p.geom_positions.forEach((pos,i)=>{const r=p.geom_rotations[i];matrix.set(r[0],r[1],r[2],pos[0],r[3],r[4],r[5],pos[1],r[6],r[7],r[8],pos[2],0,0,0,1);objects[i].matrix.copy(matrix);objects[i].matrixWorldNeedsUpdate=true});
 if(p.vertices){
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(p.vertices,3));geometry.setIndex(p.triangles);geometry.computeVertexNormals();
  if(!surface){surface=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:0xe5cf85,roughness:.78,metalness:0,side:THREE.DoubleSide}));scene.add(surface)}
  else{surface.geometry.dispose();surface.geometry=geometry}
 }
 if(p.particles.length){
  if(!particles){particles=new THREE.InstancedMesh(new THREE.SphereGeometry(p.radius*.85,10,8),new THREE.MeshStandardMaterial({roughness:.8}),p.codes.length);particles.layers.set(1);scene.add(particles)}
  const colors=[new THREE.Color(0xe9bc58),new THREE.Color(0x3698c0),new THREE.Color(0xdf8b4b)];
  p.codes.forEach((code,i)=>{dummy.position.fromArray(p.particles,3*i);dummy.updateMatrix();particles.setMatrixAt(i,dummy.matrix);particles.setColorAt(i,colors[code%3])});particles.instanceMatrix.needsUpdate=true;particles.instanceColor.needsUpdate=true;particles.computeBoundingSphere();
 }
 $('sim').textContent=p.sim_s.toFixed(3)+' s';$('wall').textContent=p.wall_s.toFixed(2)+' s';$('ratio').textContent=p.ratio.toFixed(3)+' ×';$('ratio').className=p.ratio>=1?'good':'slow';$('force').textContent=Math.hypot(...p.wrench.slice(0,3)).toFixed(3)+' N';showPieces(p.pieces||[]);
}
const states={loading:'正在初始化 GPU…',ready:'已就绪 · 点击开始计算',running:'在线计算中',paused:'已暂停 · 点击继续',completed:'本次运行已结束',stopped:'已结束',failed:'运行出错'};
async function poll(){
 let delay=0;
 try{
  const t0=performance.now();
  const response=await fetch('/state?after='+seq);
  if(response.status!==204){
   if(!response.ok)throw Error('HTTP '+response.status);
   const text=await response.text();const t1=performance.now();
   const p=JSON.parse(text);const t2=performance.now();
   if(p.generation!==undefined&&p.generation!==assetGeneration){const a=await(await fetch('/scene')).json();if(a)installAsset(a)}
   if(p.geom_positions&&assetGeneration===p.generation){applyPacket(p);const t3=performance.now();keep(timing.fetchMs,t1-t0);keep(timing.parseMs,t2-t1);keep(timing.applyMs,t3-t2);keep(timing.bytes,text.length);appliedAt=t3}
   seq=p.sequence;lastReceive=performance.now();packetCount++;lastState=p.state;
   $('connection').textContent=states[p.state]||p.state;$('connection').className='connection'+(p.state==='failed'?' error':'');
   $('start').disabled=['loading','running'].includes(p.state)||p.can_restart===false;$('pause').disabled=p.state!=='running';$('stop').disabled=!['running','paused','ready'].includes(p.state);
   $('start').textContent=['completed','stopped','failed'].includes(p.state)?'重新开始':p.state==='paused'?'继续':'开始';
   if(p.report){const success=p.report.task_evaluation?.success;$('result').textContent=p.report.exception|| (success?'单刀诊断通过':p.state==='stopped'?'用户结束':'本次诊断未通过');$('result').className=success?'good':'slow';$('storage').textContent=p.report.recording?'本次已开启原始数据记录':'本次未保存逐帧快照';}
   else $('result').textContent=p.phase==='ready'?'':p.phase||'';
  }
 }catch(e){$('connection').textContent='连接中断：'+e.message;$('connection').className='connection error';delay=500}
 // The server holds the request until a new frame, so polling again at once does not spin.
 setTimeout(poll,delay);
}
for(const command of ['start','pause','stop'])$(command).onclick=async()=>{const r=await fetch('/command/'+command,{method:'POST'});if(!r.ok)$('connection').textContent=await r.text()};
function draw(now){
 if(appliedAt!==null){keep(timing.presentMs,now-appliedAt);appliedAt=null}
 const bounds=$('stage').getBoundingClientRect();renderer.setSize(bounds.width,bounds.height,false);
 for(const v of views){const r=v.element.getBoundingClientRect(),left=r.left-bounds.left,bottom=bounds.bottom-r.bottom;v.controls.update();v.camera.aspect=r.width/r.height;v.camera.updateProjectionMatrix();renderer.setViewport(left,bottom,r.width,r.height);renderer.setScissor(left,bottom,r.width,r.height);renderer.render(scene,v.camera)}
 // Browser frame rate is never presented as simulation speed.
 window.liveDiagnostics={packets:packetCount,sequence:seq,state:lastState,stalenessMs:performance.now()-lastReceive,generation:assetGeneration,timing};
 requestAnimationFrame(draw);
}
poll();requestAnimationFrame(draw);
