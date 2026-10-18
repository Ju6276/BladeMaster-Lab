// Headless upstream CpuMlsMpmSolver driver. Metres and seconds; mass-valued
// inputs/outputs use the mass unit selected by the Python adapter.
#include <climits>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "cpu_mls_mpm_solver.h"
#include "mls_mpm_algorithm.h"

using namespace crmpm;

// Exercise upstream volume, constitutive stress and rigid impulse formulas
// directly, away from the fixed grid-mass cutoff.
void checkUnits(double massUnit) {
    if (!std::isfinite(massUnit) || massUnit <= 0)
        throw std::runtime_error("Mass unit must be positive and finite");
    const float dx = 0.005f;
    const float mass = 1100.0 * std::pow(dx / 2, 3) / massUnit;
    const float young = 6000.0 / massUnit, poisson = 0.4f;
    const float mu = young / (2 * (1 + poisson));
    const float lambda = young * poisson / ((1 + poisson) * (1 - 2 * poisson));
    const float4 particle = make_float4(0.01f, 0.01f, 0.01f, mass);
    std::vector<float4> grid(125, make_float4(0, 0, 0, 0));
    const Vec3i dims(5, 5, 5);
    mlsMpmComputeInitialGridMass<false>(Vec3f(0), 1 / dx, dims, dx * dx * dx, particle, grid.data());
    float volume = 0;
    mlsMpmComputeInitialVolume(Vec3f(0), 1 / dx, dims, dx * dx * dx, particle, grid.data(), volume);
    const Mat3x3f deformation(0.9f, 0, 0, 0, 0.85f, 0, 0, 0, 1.05f);
    Mat3x3f stress, corotatedStress, linearStress;
    computeStress(make_float4(lambda, mu, 0, 0), ParticleMaterialType::eCoRotational,
                  deformation, Mat3x3f(), volume * deformation.determinant(), volume, stress);
    computeCoRotationalStress(deformation, lambda, mu, corotatedStress);
    computeLinearElasticStress(deformation, lambda, mu, linearStress);
    float4 impulse = make_float4(0, 0, 0, 0), torque = make_float4(0, 0, 0, 0);
    addMomentumToRigidBody<false>(Vec3f(0.1f, -0.2f, 0.3f) * mass, Vec3f(0.01f, 0.02f, 0.03f), impulse, torque);
    std::cout << std::setprecision(12) << "{\"fix_corotated_dispatch\":"
              << (FIG5_FIX_COROTATED_DISPATCH ? "true" : "false") << ",\"volume_m3\":" << volume
              << ",\"stress_diagonal_pa\":[" << stress.col0.x * massUnit << ','
              << stress.col1.y * massUnit << ',' << stress.col2.z * massUnit
              << "],\"corotated_stress_diagonal_pa\":[" << corotatedStress.col0.x * massUnit << ','
              << corotatedStress.col1.y * massUnit << ',' << corotatedStress.col2.z * massUnit
              << "],\"linear_stress_diagonal_pa\":[" << linearStress.col0.x * massUnit << ','
              << linearStress.col1.y * massUnit << ',' << linearStress.col2.z * massUnit
              << "],\"impulse_ns\":[" << impulse.x * massUnit << ',' << impulse.y * massUnit << ','
              << impulse.z * massUnit << "],\"angular_impulse_nms\":[" << torque.x * massUnit << ','
              << torque.y * massUnit << ',' << torque.z * massUnit << "]}\n";
}

int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--check-units") {
            checkUnits(std::stod(argv[2]));
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--identity") {
            std::cout << FIG5_SOURCE_REVISION << '\n' << FIG5_DRIVER_REVISION << '\n'
                      << FIG5_FIX_COROTATED_DISPATCH << '\n' << FIG5_HEADER_SNAPSHOT << '\n'
                      << FIG5_DRIVER_SNAPSHOT << '\n';
            return 0;
        }
        if (argc != 4) throw std::runtime_error("Expected input.txt states.bin kick.bin");
        std::ifstream input(argv[1]);
        input.exceptions(std::ios::failbit | std::ios::badbit);
        std::ofstream output(argv[2], std::ios::binary), kickOutput(argv[3], std::ios::binary);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        kickOutput.exceptions(std::ios::failbit | std::ios::badbit);
        int n, frames, substeps;
        float dx, dt, mass, young, poisson, halfLength, halfHeight, halfThickness;
        float friction, smoothDistance, drag, cutX, kickSpeed;
        input >> n >> frames >> substeps >> dx >> dt >> mass >> young >> poisson;
        input >> halfLength >> halfHeight >> halfThickness >> friction >> smoothDistance >> drag;
        input >> cutX >> kickSpeed;
        if (n <= 0 || frames <= 0 || substeps <= 0 || dx <= 0 || dt <= 0)
            throw std::runtime_error("Invalid dimensions");
        float lo[3], hi[3];
        for (auto &value : lo) input >> value;
        for (auto &value : hi) input >> value;
        std::vector<float4> positions(n), materials(n);
        std::vector<Vec3f> velocities(n);
        std::vector<ParticleMaterialType> materialTypes(n, ParticleMaterialType::eCoRotational);
        std::vector<unsigned char> active(n, 1), left(n, 0);
        std::vector<unsigned int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        const float mu = young / (2 * (1 + poisson));
        const float lambda = young * poisson / ((1 + poisson) * (1 - 2 * poisson));
        for (int p = 0; p < n; ++p) {
            input >> positions[p].x >> positions[p].y >> positions[p].z;
            positions[p].w = mass;
            materials[p] = make_float4(lambda, mu, 0, 0);
            left[p] = positions[p].x < cutX;
        }
        ParticleData particles(n);
        particles.positionMass = positions.data();
        particles.velocity = velocities.data();
        ParticleMaterialData materialData(n);
        materialData.params0 = materials.data();
        materialData.type = materialTypes.data();

        // Dynamic type records nodal momentum exchange; eKinematic contact keeps
        // both bodies prescribed. No rigid-body integrator updates these poses.
        ShapeType shapeTypes[2] = {ShapeType::eDynamic, ShapeType::eDynamic};
        int geometryIds[2] = {0, 1}, shapeIds[2] = {0, 1};
        Vec3f shapePositions[2] = {Vec3f(0, 0, 1), Vec3f(0)};
        Vec3f shapeVelocities[2], angularVelocities[2], inverseScales[2] = {Vec3f(1), Vec3f(1)};
        const float q = std::sqrt(0.5f);
        Quat rotations[2] = {Quat(0, 0, q, q), Quat(q, 0, 0, q)};
        float4 shapeParams[2] = {make_float4(smoothDistance, drag, friction, halfThickness),
                                make_float4(0, 1, friction, 0)};
        float4 force[2] = {}, torque[2] = {}, inverseMass[2] = {}, inverseInertia[2] = {};
        ShapeData shapes;
        shapes.size = 2;
        shapes.type = shapeTypes;
        shapes.geometryIdx = geometryIds;
        shapes.position = shapePositions;
        shapes.rotation = rotations;
        shapes.invScale = inverseScales;
        shapes.linearVelocity = shapeVelocities;
        shapes.angularVelocity = angularVelocities;
        shapes.comInvMass = inverseMass;
        shapes.inertiaInv0 = inverseInertia;
        shapes.inertiaInv1 = inverseInertia;
        shapes.params0 = shapeParams;
        shapes.force = force;
        shapes.torque = torque;
        GeometryType geometryTypes[2] = {GeometryType::eQuadSlicer, GeometryType::ePlane};
        float4 geometryParams[2] = {make_float4(halfLength, halfHeight, 0, 0), make_float4(0, 0, 0, 0)};
        GeometryData geometry;
        geometry.size = 2;
        geometry.type = geometryTypes;
        geometry.params0 = geometryParams;
        GeometrySdfData sdf;
        CpuMlsMpmSolver solver(n, dx, Bounds3(Vec3f(lo[0], lo[1], lo[2]), Vec3f(hi[0], hi[1], hi[2])));
        solver.bindParticleData(particles);
        solver.bindParticleMaterials(materialData);
        solver.bindActiveMask(active.data());
        solver.setNumActiveParticles(n);
        solver.setNumShapes(2);
        solver.bindShapeIds(shapeIds);
        solver.bindShapeData(shapes);
        solver.bindGeometryData(geometry);
        solver.bindGeometrySdfData(sdf);
        solver.setGravity(Vec3f(0, 0, -9.81f));
        solver.setIntegrationStepSize(dt);
        solver.setShapeContactModel(ShapeContactModel::eKinematic);
        solver.initialize();
        solver.computeInitialData(n, indices.data());

        auto save = [&]() {
            for (int p = 0; p < n; ++p) {
                float row[6] = {positions[p].x, positions[p].y, positions[p].z,
                                velocities[p].x(), velocities[p].y(), velocities[p].z()};
                for (float value : row)
                    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite particle state");
                // Detect approach to the grid boundary before the upstream clamp.
                for (int axis = 0; axis < 3; ++axis)
                    if (row[axis] <= lo[axis] + dx || row[axis] >= hi[axis] - dx)
                        throw std::runtime_error("Particle reached grid boundary guard");
                output.write(reinterpret_cast<char *>(row), sizeof(row));
            }
            output.write(reinterpret_cast<char *>(&force[0]), 3 * sizeof(float));
            // Upstream force stores impulse on the rigid shape. Board output is
            // negated to report the impulse received by the material.
            float boardImpulse[3] = {-force[1].x, -force[1].y, -force[1].z};
            output.write(reinterpret_cast<char *>(boardImpulse), sizeof(boardImpulse));
        };
        save();
        int kicks = 0;
        for (int frame = 0; frame < frames; ++frame) {
            float px, py, pz, vx, vy, vz;
            int kick;
            input >> px >> py >> pz >> vx >> vy >> vz >> kick;
            shapePositions[0] = Vec3f(px, py, pz);
            shapeVelocities[0] = Vec3f(vx, vy, vz);
            if (kick) {
                ++kicks;
                if (kicks > 1) throw std::runtime_error("Repeated initial velocity assignment");
                for (int p = 0; p < n; ++p) {
                    if (left[p]) velocities[p] = Vec3f(kickSpeed, 0, 0);
                    float row[3] = {velocities[p].x(), velocities[p].y(), velocities[p].z()};
                    kickOutput.write(reinterpret_cast<char *>(row), sizeof(row));
                }
            }
            for (int k = 0; k < 2; ++k) force[k] = torque[k] = make_float4(0, 0, 0, 0);
            for (int step = 0; step < substeps; ++step) solver.step();
            save();
        }
        if (kicks != 1) throw std::runtime_error("Missing initial velocity assignment");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
