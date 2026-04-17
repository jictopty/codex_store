#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * 通用任意阶(p)二维曲线网格DG框架：可压Navier-Stokes方程
 * 说明：
 * 1) 这是一个“可扩展核心骨架”，重点演示任意阶DG + 曲线映射 + NS通量组织方式。
 * 2) 若要用于真实算例，需要替换：网格读入、基函数/矩阵生成、边界条件、时间推进(SSPRK/IMEX)等。
 */

enum { NVAR = 4 }; /* rho, rho*u, rho*v, E */

typedef struct {
    int p;      /* 多项式阶数 */
    int nq;     /* 每单元体积点数 */
    int nfp;    /* 每面点数 */
    int nface;  /* 面数(三角形=3,四边形=4) */

    /* 参考元微分矩阵: d()/dr, d()/ds */
    double *Dr; /* [nq x nq] */
    double *Ds; /* [nq x nq] */

    /* 体积点插值到每个面的矩阵 */
    double *Vf; /* [nface*nfp x nq] */

    /* lifting矩阵: 将面通量lift到体积 */
    double *LIFT; /* [nq x (nface*nfp)] */

    /* 每个面点在参考元上的法向(仅方向，长度由几何缩放) */
    double *nr_ref; /* [nface*nfp] */
    double *ns_ref; /* [nface*nfp] */
} DGSpace;

typedef struct {
    int nelem;

    /* 曲线网格几何量(每单元每体积点) */
    double *rx, *sx; /* x_r^{-1}, x_s^{-1}相关度量 */
    double *ry, *sy;
    double *J;       /* Jacobian */

    /* 面几何缩放和物理法向(每单元每面点) */
    double *nx, *ny; /* 物理单位法向 */
    double *sJ;      /* face Jacobian (|∂x/∂ξ_face|) */

    /* 邻接：对每个(单元,局部面点)给出邻单元和邻面点索引 */
    int *nbr_elem; /* -1 表示边界 */
    int *nbr_fpt;  /* 邻面点(扁平索引) */
} CurviMesh;

typedef struct {
    double gamma;
    double mu;   /* 动力黏性系数（此处视作常数） */
    double Pr;   /* Prandtl */
    double Rgas; /* 气体常数 */
} NSParams;

static inline int id_u(int e, int q, int v, int nq) {
    return (e * nq + q) * NVAR + v;
}

static inline int id_m(int e, int q, int nq) { return e * nq + q; }

static void cons_to_prim(const double U[NVAR], const NSParams *p, double *rho,
                         double *u, double *v, double *P, double *T, double *H) {
    *rho = U[0];
    *u = U[1] / U[0];
    *v = U[2] / U[0];
    double E = U[3];
    double vel2 = (*u) * (*u) + (*v) * (*v);
    *P = (p->gamma - 1.0) * (E - 0.5 * (*rho) * vel2);
    *T = *P / ((*rho) * p->Rgas);
    *H = (E + *P) / (*rho);
}

static void inviscid_flux(const double U[NVAR], const NSParams *p, double Fx[NVAR],
                          double Fy[NVAR]) {
    double rho, u, v, P, T, H;
    (void)T;
    cons_to_prim(U, p, &rho, &u, &v, &P, &T, &H);

    Fx[0] = rho * u;
    Fx[1] = rho * u * u + P;
    Fx[2] = rho * u * v;
    Fx[3] = rho * u * H;

    Fy[0] = rho * v;
    Fy[1] = rho * u * v;
    Fy[2] = rho * v * v + P;
    Fy[3] = rho * v * H;
}

static double max_wave_speed_n(const double U[NVAR], const NSParams *p, double nx,
                               double ny) {
    double rho, u, v, P, T, H;
    (void)T;
    (void)H;
    cons_to_prim(U, p, &rho, &u, &v, &P, &T, &H);
    double a = sqrt(p->gamma * P / rho);
    double un = u * nx + v * ny;
    return fabs(un) + a;
}

static void rusanov_flux(const double UL[NVAR], const double UR[NVAR],
                         const NSParams *p, double nx, double ny,
                         double Fn[NVAR]) {
    double FLx[NVAR], FLy[NVAR], FRx[NVAR], FRy[NVAR];
    inviscid_flux(UL, p, FLx, FLy);
    inviscid_flux(UR, p, FRx, FRy);

    double smaxL = max_wave_speed_n(UL, p, nx, ny);
    double smaxR = max_wave_speed_n(UR, p, nx, ny);
    double lam = smaxL > smaxR ? smaxL : smaxR;

    for (int m = 0; m < NVAR; ++m) {
        double fnL = FLx[m] * nx + FLy[m] * ny;
        double fnR = FRx[m] * nx + FRy[m] * ny;
        Fn[m] = 0.5 * (fnL + fnR) - 0.5 * lam * (UR[m] - UL[m]);
    }
}

/* 简单绝热无滑移壁面：镜像速度 */
static void wall_state(const double Uin[NVAR], const NSParams *p, double nx,
                       double ny, double Ughost[NVAR]) {
    double rho, u, v, P, T, H;
    (void)T;
    (void)H;
    cons_to_prim(Uin, p, &rho, &u, &v, &P, &T, &H);

    double un = u * nx + v * ny;
    double ut = -u * ny + v * nx;
    double un_g = -un;
    double ut_g = ut;
    double ug = un_g * nx - ut_g * ny;
    double vg = un_g * ny + ut_g * nx;

    Ughost[0] = rho;
    Ughost[1] = rho * ug;
    Ughost[2] = rho * vg;
    Ughost[3] = Uin[3];
}

/* 构造∂U/∂x, ∂U/∂y (每单元每点每变量) */
static void compute_physical_gradients(const DGSpace *sp, const CurviMesh *mesh,
                                       const double *U, double *Ux, double *Uy) {
    int nq = sp->nq;
    for (int e = 0; e < mesh->nelem; ++e) {
        for (int q = 0; q < nq; ++q) {
            double rx = mesh->rx[id_m(e, q, nq)], sx = mesh->sx[id_m(e, q, nq)];
            double ry = mesh->ry[id_m(e, q, nq)], sy = mesh->sy[id_m(e, q, nq)];
            for (int m = 0; m < NVAR; ++m) {
                double Ur = 0.0, Us = 0.0;
                for (int j = 0; j < nq; ++j) {
                    double Uj = U[id_u(e, j, m, nq)];
                    Ur += sp->Dr[q * nq + j] * Uj;
                    Us += sp->Ds[q * nq + j] * Uj;
                }
                Ux[id_u(e, q, m, nq)] = rx * Ur + sx * Us;
                Uy[id_u(e, q, m, nq)] = ry * Ur + sy * Us;
            }
        }
    }
}

/* 体积分项: -∇·F(U,∇U)，仅示例性包含常黏度应力与导热 */
static void volume_residual(const DGSpace *sp, const CurviMesh *mesh,
                            const NSParams *p, const double *U,
                            const double *Ux, const double *Uy, double *RHS) {
    int nq = sp->nq;
    (void)RHS;
    double kappa = p->mu * p->gamma * p->Rgas / ((p->gamma - 1.0) * p->Pr);

    for (int e = 0; e < mesh->nelem; ++e) {
        for (int q = 0; q < nq; ++q) {
            double rho, u, v, P, T, H;
            (void)H;
            cons_to_prim(&U[id_u(e, q, 0, nq)], p, &rho, &u, &v, &P, &T, &H);

            /* 由保守量梯度近似速度与温度梯度 (简化版) */
            double rhox = Ux[id_u(e, q, 0, nq)], rhoy = Uy[id_u(e, q, 0, nq)];
            double rhoux = Ux[id_u(e, q, 1, nq)], rhouy = Uy[id_u(e, q, 1, nq)];
            double rhovx = Ux[id_u(e, q, 2, nq)], rhovy = Uy[id_u(e, q, 2, nq)];
            double Ex = Ux[id_u(e, q, 3, nq)], Ey = Uy[id_u(e, q, 3, nq)];

            double ux = (rhoux * rho - U[id_u(e, q, 1, nq)] * rhox) / (rho * rho);
            double uy = (rhouy * rho - U[id_u(e, q, 1, nq)] * rhoy) / (rho * rho);
            double vx = (rhovx * rho - U[id_u(e, q, 2, nq)] * rhox) / (rho * rho);
            double vy = (rhovy * rho - U[id_u(e, q, 2, nq)] * rhoy) / (rho * rho);

            double px = (p->gamma - 1.0) *
                        (Ex - 0.5 * (rhox * (u * u + v * v) +
                                     2.0 * rho * (u * ux + v * vx)));
            double py = (p->gamma - 1.0) *
                        (Ey - 0.5 * (rhoy * (u * u + v * v) +
                                     2.0 * rho * (u * uy + v * vy)));
            double Tx = (px * rho - P * rhox) / (rho * rho * p->Rgas);
            double Ty = (py * rho - P * rhoy) / (rho * rho * p->Rgas);

            double divu = ux + vy;
            double tauxx = p->mu * (2.0 * ux - (2.0 / 3.0) * divu);
            double tauyy = p->mu * (2.0 * vy - (2.0 / 3.0) * divu);
            double tauxy = p->mu * (uy + vx);

            /* 强形式DG体积项: 
             * RHS += -(Dr,Ds)·(metric * flux)
             * 这里为简化直接点值近似，真实代码应做矩阵-向量收缩。
             */
            (void)tauxx;
            (void)tauyy;
            (void)tauxy;
            (void)kappa;
            (void)Tx;
            (void)Ty;
            /* 留空：结构已就位，具体算子收缩依赖参考元积分类与矩阵装配 */
        }
    }
}

static void surface_residual(const DGSpace *sp, const CurviMesh *mesh,
                             const NSParams *p, const double *U, double *RHS) {
    int nq = sp->nq;
    int nsp = sp->nface * sp->nfp;

    double *Uf = (double *)calloc((size_t)mesh->nelem * nsp * NVAR, sizeof(double));
    double *Fhat = (double *)calloc((size_t)mesh->nelem * nsp * NVAR, sizeof(double));

    if (!Uf || !Fhat) {
        fprintf(stderr, "allocation failed in surface_residual\n");
        free(Uf);
        free(Fhat);
        return;
    }

    /* 插值到面点 */
    for (int e = 0; e < mesh->nelem; ++e) {
        for (int fp = 0; fp < nsp; ++fp) {
            for (int m = 0; m < NVAR; ++m) {
                double v = 0.0;
                for (int q = 0; q < nq; ++q) {
                    v += sp->Vf[fp * nq + q] * U[id_u(e, q, m, nq)];
                }
                Uf[(e * nsp + fp) * NVAR + m] = v;
            }
        }
    }

    /* 数值通量 */
    for (int e = 0; e < mesh->nelem; ++e) {
        for (int fp = 0; fp < nsp; ++fp) {
            double UL[NVAR], UR[NVAR], Fn[NVAR];
            for (int m = 0; m < NVAR; ++m) UL[m] = Uf[(e * nsp + fp) * NVAR + m];

            int gid = e * nsp + fp;
            int en = mesh->nbr_elem[gid];
            int fpn = mesh->nbr_fpt[gid];
            double nx = mesh->nx[gid], ny = mesh->ny[gid];

            if (en >= 0) {
                for (int m = 0; m < NVAR; ++m)
                    UR[m] = Uf[(en * nsp + fpn) * NVAR + m];
            } else {
                wall_state(UL, p, nx, ny, UR);
            }

            rusanov_flux(UL, UR, p, nx, ny, Fn);
            for (int m = 0; m < NVAR; ++m)
                Fhat[(e * nsp + fp) * NVAR + m] = Fn[m] * mesh->sJ[gid];
        }
    }

    /* lift到体积RHS */
    for (int e = 0; e < mesh->nelem; ++e) {
        for (int q = 0; q < nq; ++q) {
            for (int m = 0; m < NVAR; ++m) {
                double acc = 0.0;
                for (int fp = 0; fp < nsp; ++fp) {
                    acc += sp->LIFT[q * nsp + fp] * Fhat[(e * nsp + fp) * NVAR + m];
                }
                RHS[id_u(e, q, m, nq)] -= acc;
            }
        }
    }

    free(Uf);
    free(Fhat);
}

void dg_ns_rhs(const DGSpace *sp, const CurviMesh *mesh, const NSParams *p,
               const double *U, double *RHS) {
    size_t nall = (size_t)mesh->nelem * sp->nq * NVAR;
    memset(RHS, 0, nall * sizeof(double));

    double *Ux = (double *)calloc(nall, sizeof(double));
    double *Uy = (double *)calloc(nall, sizeof(double));

    if (!Ux || !Uy) {
        fprintf(stderr, "allocation failed in dg_ns_rhs\n");
        free(Ux);
        free(Uy);
        return;
    }

    compute_physical_gradients(sp, mesh, U, Ux, Uy);
    volume_residual(sp, mesh, p, U, Ux, Uy, RHS);
    surface_residual(sp, mesh, p, U, RHS);

    free(Ux);
    free(Uy);
}

int main(void) {
    printf("DG NS curvilinear solver skeleton is ready.\n");
    printf("Next step: load mesh/operators and run RK time integration.\n");
    return 0;
}
