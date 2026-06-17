#pragma once
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#ifdef _OPENMP
#include <omp.h>
#endif

struct MagneticNull {
    double x, y, z;
    int type; // 0 = Radial (3 real eigenvalues), 1 = Spiral (1 real, 2 complex)
};

namespace null_finder {

// Helper: Evaluate trilinear interpolation and its derivatives
inline void trilinear_eval(const double c[8], double u, double v, double w, 
                           double& val, double& du, double& dv, double& dw) {
    double inv_u = 1.0 - u, inv_v = 1.0 - v, inv_w = 1.0 - w;
    
    val = c[0]*inv_u*inv_v*inv_w + c[1]*u*inv_v*inv_w +
          c[2]*inv_u*v*inv_w     + c[3]*u*v*inv_w +
          c[4]*inv_u*inv_v*w     + c[5]*u*inv_v*w +
          c[6]*inv_u*v*w         + c[7]*u*v*w;

    du  = -c[0]*inv_v*inv_w + c[1]*inv_v*inv_w 
          -c[2]*v*inv_w     + c[3]*v*inv_w 
          -c[4]*inv_v*w     + c[5]*inv_v*w 
          -c[6]*v*w         + c[7]*v*w;

    dv  = -c[0]*inv_u*inv_w - c[1]*u*inv_w 
          +c[2]*inv_u*inv_w + c[3]*u*inv_w 
          -c[4]*inv_u*w     - c[5]*u*w 
          +c[6]*inv_u*w     + c[7]*u*w;

    dw  = -c[0]*inv_u*inv_v - c[1]*u*inv_v 
          -c[2]*inv_u*v     - c[3]*u*v 
          +c[4]*inv_u*inv_v + c[5]*u*inv_v 
          +c[6]*inv_u*v     + c[7]*u*v;
}

inline std::vector<MagneticNull> find_nulls(int N, double L, 
                                            const double* Bx, const double* By, const double* Bz) {
    std::vector<MagneticNull> global_nulls;
    const double dx = L / N;

    auto get_idx = [N](int i, int j, int k) {
        return ((size_t)(i % N) * N + (j % N)) * N + (k % N);
    };

    #pragma omp parallel
    {
        std::vector<MagneticNull> local_nulls;

        #pragma omp for collapse(3) schedule(static)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < N; ++k) {
                    
                    // Fetch 8 corners of the voxel
                    double cx[8], cy[8], cz[8];
                    int idx = 0;
                    for(int dk=0; dk<=1; ++dk) {
                        for(int dj=0; dj<=1; ++dj) {
                            for(int di=0; di<=1; ++di) {
                                size_t flat_idx = get_idx(i+di, j+dj, k+dk);
                                cx[idx] = Bx[flat_idx];
                                cy[idx] = By[flat_idx];
                                cz[idx] = Bz[flat_idx];
                                idx++;
                            }
                        }
                    }

                    // 1. Fast Bounding Box Reject
                    double min_bx = cx[0], max_bx = cx[0];
                    double min_by = cy[0], max_by = cy[0];
                    double min_bz = cz[0], max_bz = cz[0];
                    double max_b_sq = 0.0; // <-- ADD THIS

                    for(int m=1; m<8; ++m) {
                        if(cx[m] < min_bx) min_bx = cx[m]; if(cx[m] > max_bx) max_bx = cx[m];
                        if(cy[m] < min_by) min_by = cy[m]; if(cy[m] > max_by) max_by = cy[m];
                        if(cz[m] < min_bz) min_bz = cz[m]; if(cz[m] > max_bz) max_bz = cz[m];
                    }
                    if (min_bx > 0 || max_bx < 0 || 
                        min_by > 0 || max_by < 0 || 
                        min_bz > 0 || max_bz < 0) continue;
                        
                    // <-- ADD THIS NOISE FLOOR REJECT -->
                    // Calculate max B^2 in this cell
                    for(int m=0; m<8; ++m) {
                        double b_sq = cx[m]*cx[m] + cy[m]*cy[m] + cz[m]*cz[m];
                        if (b_sq > max_b_sq) max_b_sq = b_sq;
                    }
                    // If the field magnitude is essentially machine noise, reject it.
                    // 1e-10 for B^2 means |B| ~ 1e-5. True fields are ~O(0.1 to 1.0).
                    if (max_b_sq < 1e-10) continue; 
                    // <-------------------------------->

                        
                    // 2. Newton-Raphson Iteration
                    Eigen::Vector3d u(0.5, 0.5, 0.5); // Start at cell center
                    bool converged = false;
                    
                    for (int iter = 0; iter < 10; ++iter) {
                        double bx, dbx_du, dbx_dv, dbx_dw;
                        double by, dby_du, dby_dv, dby_dw;
                        double bz, dbz_du, dbz_dv, dbz_dw;
                        
                        trilinear_eval(cx, u.x(), u.y(), u.z(), bx, dbx_du, dbx_dv, dbx_dw);
                        trilinear_eval(cy, u.x(), u.y(), u.z(), by, dby_du, dby_dv, dby_dw);
                        trilinear_eval(cz, u.x(), u.y(), u.z(), bz, dbz_du, dbz_dv, dbz_dw);

                        Eigen::Vector3d B_val(bx, by, bz);
                        Eigen::Matrix3d J_uvw;
                        J_uvw << dbx_du, dbx_dv, dbx_dw,
                                 dby_du, dby_dv, dby_dw,
                                 dbz_du, dbz_dv, dbz_dw;

                        Eigen::Vector3d delta = J_uvw.fullPivLu().solve(B_val);
                        u -= delta;

                        if (delta.norm() < 1e-6) {
                            converged = true;
                            break;
                        }
                    }

                    // 3. Verify it stayed within this specific voxel [0, 1) to prevent double counting
                    if (converged && 
                        u.x() >= -1e-8 && u.x() < 1.0 - 1e-8 &&
                        u.y() >= -1e-8 && u.y() < 1.0 - 1e-8 &&
                        u.z() >= -1e-8 && u.z() < 1.0 - 1e-8) 
                    {
                        // Calculate physical Jacobian (J_uvw / dx)
                        double bx, dbx_du, dbx_dv, dbx_dw;
                        double by, dby_du, dby_dv, dby_dw;
                        double bz, dbz_du, dbz_dv, dbz_dw;
                        trilinear_eval(cx, u.x(), u.y(), u.z(), bx, dbx_du, dbx_dv, dbx_dw);
                        trilinear_eval(cy, u.x(), u.y(), u.z(), by, dby_du, dby_dv, dby_dw);
                        trilinear_eval(cz, u.x(), u.y(), u.z(), bz, dbz_du, dbz_dv, dbz_dw);

                        Eigen::Matrix3d J_phys;
                        J_phys << dbx_du, dbx_dv, dbx_dw,
                                  dby_du, dby_dv, dby_dw,
                                  dbz_du, dbz_dv, dbz_dw;
                        J_phys /= dx;

                        // 4. Eigenvalue Classification
                        Eigen::EigenSolver<Eigen::Matrix3d> solver(J_phys);
                        auto evals = solver.eigenvalues();
                        int complex_count = 0;
                        for(int e=0; e<3; ++e) {
                            if (std::abs(evals[e].imag()) > 1e-8) complex_count++;
                        }
                        int type = (complex_count > 0) ? 1 : 0; // 1 = Spiral, 0 = Radial

                        // Store physical coordinates
                        MagneticNull null_pt;
                        null_pt.x = -L/2.0 + (i + u.x()) * dx;
                        null_pt.y = -L/2.0 + (j + u.y()) * dx;
                        null_pt.z = -L/2.0 + (k + u.z()) * dx;
                        null_pt.type = type;
                        
                        local_nulls.push_back(null_pt);
                    }
                }
            }
        }

        #pragma omp critical
        {
            global_nulls.insert(global_nulls.end(), local_nulls.begin(), local_nulls.end());
        }
    }

    return global_nulls;
}

} // namespace null_finder