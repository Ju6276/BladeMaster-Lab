import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
const $ = id => document.getElementById(id);
// Page files the open page was loaded with; a restarted server with new files reloads it.
let pageVersion=null;
// One colour per connected piece; piece 0 keeps the material colour of each particle (asset.particle_rgb).
const pieceColors=new Map();let materialColors=null,surfaceColor=new THREE.Color(0xe5cf85);
function pieceColor(piece,i){if(piece===0&&materialColors)return materialColors[i];if(!pieceColors.has(piece))pieceColors.set(piece,piece===0?new THREE.Color(0xe9bc58):new THREE.Color().setHSL((0.55+piece*0.618034)%1,.62,.52));return pieceColors.get(piece)}
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
let sceneMetadata=null;
const timing = {fetchMs: [], parseMs: [], applyMs: [], bytes: [], presentMs: [],
 renderMs: [], renderIntervalMs: [], packetIntervalMs: []};
let pendingPacket = null, previousDraw = null, previousPacketDraw = null;
let renderedPacket = null, renderedPackets = 0;
function keep(list,value){list.push(value);if(list.length>200)list.shift()}
const matrix=new THREE.Matrix4(),dummy=new THREE.Object3D();
function showSceneMetadata(asset) {
 const material = asset.material_grid_spacing_m, surface = asset.surface_spacing_m;
 const mm = value => Number.isFinite(value) ? Number((value * 1000).toPrecision(6)) + ' mm' : '—';
 $('material-spacing').textContent = material === null ? '未启用材料' : '材料网格间距 ' + mm(material);
 $('surface-spacing').textContent = material === null ? '未启用材料'
  : surface === null ? '未启用重建' : '表面网格间距 ' + mm(surface);
 const revision = asset.build_revision;
 const dirty = revision?.dirty === true ? ' · 含未提交修改'
  : revision?.dirty === false ? '' : ' · 工作区状态未知';
 $('build').title = '运行版本（进程启动时）：' + (revision?.commit || '未知') + dirty;
 sceneMetadata = {
  build_revision: revision ?? null,
  material_grid_spacing_m: material ?? null,
  surface_spacing_m: surface ?? null
 };
}
function installAsset(asset){
 showSceneMetadata(asset);
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
 // Material colours are sRGB, as in the offline renderer.
 const srgb=(r,g,b)=>new THREE.Color().setRGB(r,g,b,THREE.SRGBColorSpace),rgb=asset.particle_rgb;
 materialColors=rgb?Array.from({length:rgb.length/3},(_,i)=>srgb(rgb[3*i],rgb[3*i+1],rgb[3*i+2])):null;
 surfaceColor=asset.surface_rgb?srgb(...asset.surface_rgb):new THREE.Color(0xe5cf85);pieceColors.clear();
 const origin=new THREE.Vector3(...asset.origin);
 // Paper Fig. 1 view: in front of the arm, looking back along its reach, slightly from above.
 const base=new THREE.Vector3(...(asset.base||[0,0,0])),reach=origin.clone().sub(base).setZ(0),front=reach.clone().normalize();
 const whole=base.clone().addScaledVector(reach,.5).setZ(.52);
 views.forEach((v,i)=>{v.controls.target.copy(i?origin.clone().add(new THREE.Vector3(0,0,.025)):whole);v.camera.position.copy(v.controls.target).add(i?front.clone().multiplyScalar(.24).setZ(.19):front.clone().multiplyScalar(1.2).setZ(.85));v.controls.update()});
 assetGeneration=asset.generation;
}
function applyPacket(p){
 p.geom_positions.forEach((pos,i)=>{const r=p.geom_rotations[i];matrix.set(r[0],r[1],r[2],pos[0],r[3],r[4],r[5],pos[1],r[6],r[7],r[8],pos[2],0,0,0,1);objects[i].matrix.copy(matrix);objects[i].matrixWorldNeedsUpdate=true});
 if(p.vertices){
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(p.vertices,3));geometry.setIndex(p.triangles);geometry.computeVertexNormals();
  if(!surface){surface=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:surfaceColor,roughness:.78,metalness:0,side:THREE.DoubleSide}));scene.add(surface)}
  else{surface.geometry.dispose();surface.geometry=geometry}
 }
 if(p.particles.length){
  if(!particles){particles=new THREE.InstancedMesh(new THREE.SphereGeometry(p.radius*.85,10,8),new THREE.MeshStandardMaterial({roughness:.8}),p.codes.length);particles.layers.set(1);scene.add(particles)}
  p.codes.forEach((code,i)=>{dummy.position.fromArray(p.particles,3*i);dummy.updateMatrix();particles.setMatrixAt(i,dummy.matrix);particles.setColorAt(i,pieceColor(p.piece.length?p.piece[i]:0,i))});particles.instanceMatrix.needsUpdate=true;particles.instanceColor.needsUpdate=true;particles.computeBoundingSphere();
 }
 $('sim').textContent=p.sim_s.toFixed(3)+' s';$('wall').textContent=p.wall_s.toFixed(2)+' s';$('ratio').textContent=p.ratio.toFixed(3)+' ×';$('ratio').className=p.ratio>=1?'good':'slow';$('force').textContent=Math.hypot(...p.wrench.slice(0,3)).toFixed(3)+' N';showPieces(p.pieces||[]);
}
const states={loading:'正在初始化 GPU…',ready:'已就绪 · 点击开始计算',running:'在线计算中',paused:'已暂停 · 点击继续',completed:'本次运行已结束',stopped:'已结束',failed:'运行出错'};
// The server holds a poll for at most 0.5 s; a stalled connection is aborted and retried.
const POLL_TIMEOUT_MS=5000;
// The static robot meshes are much larger than a state frame, especially over SSH.
const SCENE_TIMEOUT_MS=60000;
async function fetchJson(url, timeoutMs) {
 const abort = new AbortController();
 const timer = setTimeout(() => abort.abort(), timeoutMs);
 try {
  const response = await fetch(url, {signal: abort.signal});
  if (response.status === 204) return null;
  if (!response.ok) throw Error('HTTP ' + response.status);
  const text = await response.text();
  const receivedAt = performance.now();
  return {data: JSON.parse(text), receivedAt, bytes: text.length};
 } finally {
  clearTimeout(timer);
 }
}
function showStatus(packet) {
 const state = packet.state;
 $('connection').textContent = states[state] || state;
 $('connection').className = 'connection' + (state === 'failed' ? ' error' : '');
 $('start').disabled = ['loading', 'running'].includes(state) || packet.can_restart === false;
 $('pause').disabled = state !== 'running';
 $('stop').disabled = !['running', 'paused', 'ready'].includes(state);
 $('start').textContent = ['completed', 'stopped', 'failed'].includes(state)
  ? '重新开始' : state === 'paused' ? '继续' : '开始';
 if (packet.report) {
  const report = packet.report;
  const success = report.task_evaluation?.success;
  let result = '本次运行已结束';
  if (success === true) result = '切割流程通过';
  else if (state === 'stopped') result = '用户结束';
  else if (success === false) result = '切割流程未完成';
  $('result').textContent = report.exception || result;
  $('result').className = success ? 'good' : 'slow';
  $('storage').textContent = report.recording ? '本次已开启原始数据记录' : '本次未保存逐帧快照';
 } else {
  $('result').textContent = packet.phase === 'ready' ? '' : packet.phase || '';
 }
}

async function poll() {
 let delay = 0;
 try {
  const begin = performance.now();
  const response = await fetchJson('/state?after=' + seq, POLL_TIMEOUT_MS);
  if (response !== null) {
   const packet = response.data;
   const parsedAt = performance.now();
   if (packet.generation !== undefined && packet.generation !== assetGeneration) {
    $('connection').textContent = '正在加载机器人场景…';
    const asset = await fetchJson('/scene', SCENE_TIMEOUT_MS);
    if (asset?.data) {
     if (pageVersion === null) pageVersion = asset.data.web_version;
     else if (asset.data.web_version !== pageVersion) {
      location.reload();
      return;
     }
     installAsset(asset.data);
    }
   }
   if (packet.geom_positions && assetGeneration === packet.generation) {
    const beginApply = performance.now();
    applyPacket(packet);
    const appliedAt = performance.now();
    keep(timing.fetchMs, response.receivedAt - begin);
    keep(timing.parseMs, parsedAt - response.receivedAt);
    keep(timing.applyMs, appliedAt - beginApply);
    keep(timing.bytes, response.bytes);
    pendingPacket = {
     appliedAt, sequence: packet.sequence, inputSequence: packet.input_seq, simS: packet.sim_s
    };
   }
   seq = packet.sequence;
   lastReceive = performance.now();
   packetCount++;
   lastState = packet.state;
   showStatus(packet);
  }
 } catch (error) {
  $('connection').textContent = error.name === 'AbortError'
   ? '连接超时，重试中' : '连接中断：' + error.message;
  $('connection').className = 'connection error';
  delay = 500;
 }
 // The server holds the request until a new frame, so polling again at once does not spin.
 setTimeout(poll, delay);
}
for(const command of ['start','pause','stop'])$(command).onclick=async()=>{const r=await fetch('/command/'+command,{method:'POST'});if(!r.ok)$('connection').textContent=await r.text()};
function draw() {
 const begin = performance.now();
 if (previousDraw !== null) keep(timing.renderIntervalMs, begin - previousDraw);
 previousDraw = begin;
 const bounds = $('stage').getBoundingClientRect();
 renderer.setSize(bounds.width, bounds.height, false);
 for (const view of views) {
  const rect = view.element.getBoundingClientRect();
  view.controls.update();
  view.camera.aspect = rect.width / rect.height;
  view.camera.updateProjectionMatrix();
  renderer.setViewport(rect.left - bounds.left, bounds.bottom - rect.bottom, rect.width, rect.height);
  renderer.setScissor(rect.left - bounds.left, bounds.bottom - rect.bottom, rect.width, rect.height);
  renderer.render(scene, view.camera);
 }
 const submitted = performance.now();
 keep(timing.renderMs, submitted - begin);
 if (pendingPacket !== null) {
  keep(timing.presentMs, submitted - pendingPacket.appliedAt);
  if (previousPacketDraw !== null) keep(timing.packetIntervalMs, submitted - previousPacketDraw);
  previousPacketDraw = submitted;
  renderedPacket = {...pendingPacket, submittedAt: submitted};
  renderedPackets++;
  pendingPacket = null;
 }
 // Submission to WebGL is observable here; the physical display scanout is not.
 window.liveDiagnostics = {
  packets: packetCount, renderedPackets, renderedPacket, sequence: seq,
  state: lastState, stalenessMs: submitted - lastReceive, generation: assetGeneration,
  scene: sceneMetadata, timing
 };
 requestAnimationFrame(draw);
}
poll();requestAnimationFrame(draw);
