%% MAIN_EGTimeBE2: Enriched Galerkin Method for Time-Dependent Stokes (2D, BE)
%
% Solves the time-dependent Stokes equations in a 2-dimensional domain:
%
%        du/dt - mu*Delta(u) + grad(p) = f, in Omega
%                               div(u) = 0, in Omega
%
% using Backward Euler in time and the Enriched Galerkin (EG) method in
% space (continuous P1 + discontinuous bubble velocity x - x_T per
% element, discontinuous P0 pressure).
%
% TestType controls how the body force f and the (u,v)/dt time-mass term
% are assembled:
%   TestType = 1: standard EG. Both the body force and the (u,v)/dt
%                 time-mass term are assembled directly, elementwise.
%   TestType = 2: pressure-robust body force only. The body force is
%                 assembled through the velocity reconstruction operator,
%                 while the time-mass term is left unmodified, exactly as
%                 in TestType 1.
%   TestType = 3: pressure-robust body force AND time-mass term. Both the
%                 body force and the (u,v)/dt time-mass term are assembled
%                 through the velocity reconstruction operator. Changing
%                 the body force alone (TestType 2) is not enough once a
%                 time derivative is present, the time-mass term needs
%                 the same treatment.
%
% Dependencies (from iFEM by L. Chen):
%   auxstructure.m, squaremesh.m, gradbasis.m
%
% References:
%   S.-Y. Yi, X. Hu, S. Lee, and J. H. Adler, "An enriched Galerkin method
%     for the Stokes equations," Computers & Mathematics with Applications, 2022.
%   X. Hu, S. Lee, L. Mu, and S.-Y. Yi, "Pressure-robust enriched Galerkin
%     methods for the Stokes equations," Journal of Computational and
%     Applied Mathematics, 2024.
%   S. Lee and L. Mu, "Fully discrete analysis of pressure-robust enriched
%     Galerkin methods for the time-dependent Stokes equations," submitted,
%     arXiv: 2608.02913, 2026.
%
% Authors: Seulip Lee and Lin Mu
%
clear

%% Preliminaries

Num = 16;
rho1 = 3; dt = 1/Num; T_end = 1; time_it = T_end/dt;
TestType = 3; % 1: standard / 2: PR body force only / 3: PR body force + mass term
assert(ismember(TestType,[1 2 3]), 'TestType must be 1, 2, or 3.');
pde = Stokes_Time2; mu_fun = pde.mu;

[node,elem] = squaremesh([0,1,0,1],1/Num);
T = auxstructure(elem);
[Dphi,area] = gradbasis(node,elem);
xT = (node(elem(:,1),:)+node(elem(:,2),:)+node(elem(:,3),:))/3;
xE = (node(T.edge(:,1),:)+node(T.edge(:,2),:))/2;

NT = size(elem,1); NE = size(T.edge,1); NO = size(node,1);

DoF_u = (2*NO+NT); DoF_p = NT; DoF = DoF_u + DoF_p;

mu_elem = mu_fun(xT);

[lambda,weight] = quadpts(9); phi = lambda; nQuad = size(lambda,1);

%% Assemble Element Terms

A = sparse(DoF,DoF);
TID = (1:NT)';
localC = [2 1 1; 1 2 1; 1 1 2]/4;

for i = 1:3
    for j = 1:3
        Acc = mu_elem.*dot(Dphi(:,:,i),Dphi(:,:,j),2).*area;
        A = A + sparse([elem(:,i);NO+elem(:,i)],[elem(:,j);NO+elem(:,j)],[Acc;Acc],DoF,DoF);
    end
    Acp = Dphi(:,:,i).*area;
    Acd = mu_elem.*Acp;
    ii = [elem(:,i),NO+elem(:,i),2*NO+TID,2*NO+TID,...
        elem(:,i),NO+elem(:,i),2*NO+NT+TID,2*NO+NT+TID];
    jj = [2*NO+TID,2*NO+TID,elem(:,i),NO+elem(:,i),...
        2*NO+NT+TID,2*NO+NT+TID,elem(:,i),NO+elem(:,i)];
    ss = [Acd,Acd,-Acp,-Acp];
    A = A + sparse(ii,jj,ss,DoF,DoF);
end

Adp = 2*area;
Add = mu_elem.*Adp;
A = A + sparse([2*NO+TID,2*NO+TID,2*NO+NT+TID], ...
    [2*NO+TID,2*NO+NT+TID,2*NO+TID],[Add,-Adp,-Adp],DoF,DoF);

%% Assemble Interior Edge Terms

B = sparse(NT,NT);

bdEdge = find(T.edge2elem(:,1)==T.edge2elem(:,2));
inEdge = setdiff((1:NE)',bdEdge);

xEin = (node(T.edge(inEdge,1),:)+node(T.edge(inEdge,2),:))/2;

lenginEdge = sqrt(sum((node(T.edge(inEdge,1),:)-node(T.edge(inEdge,2),:)).^2,2));
normVecin = [(node(T.edge(inEdge,2),2)-node(T.edge(inEdge,1),2)),...
             (node(T.edge(inEdge,1),1)-node(T.edge(inEdge,2),1))];
normVecin = normVecin./lenginEdge;

TL = double(T.edge2elem(inEdge,1));
TR = double(T.edge2elem(inEdge,2));
mu_inEdge = (mu_elem(TL)+mu_elem(TR))/2;

tmpL = dot(normVecin,xT(TR,:)-xT(TL,:),2);
tmpIdx = tmpL<0;
normVecin(tmpIdx,:) = -normVecin(tmpIdx,:);

DphiL = Dphi(TL,:,:); DphiR = Dphi(TR,:,:);
JumpL = xEin - xT(TL,:); JumpR = xEin - xT(TR,:);

AcdLL = sum(DphiL.*normVecin,2).*JumpL.*lenginEdge;
AcdRL = sum(DphiR.*normVecin,2).*JumpL.*lenginEdge;
Add0L = dot(normVecin,JumpL,2).*lenginEdge;

AcdRR = -sum(DphiR.*normVecin,2).*JumpR.*lenginEdge;
AcdLR = -sum(DphiL.*normVecin,2).*JumpR.*lenginEdge;
Add0R = -dot(normVecin,JumpR,2).*lenginEdge;

AcdLL = permute(AcdLL,[1 3 2]);
AcdRL = permute(AcdRL,[1 3 2]);
AcdRR = permute(AcdRR,[1 3 2]);
AcdLR = permute(AcdLR,[1 3 2]);

AcdLL = [AcdLL(:,:,1),AcdLL(:,:,2)];
AcdRL = [AcdRL(:,:,1),AcdRL(:,:,2)];
AcdRR = [AcdRR(:,:,1),AcdRR(:,:,2)];
AcdLR = [AcdLR(:,:,1),AcdLR(:,:,2)];

valL = [AcdLL,AcdRL,Add0L,Add0L]/2;
valR = [AcdLR,AcdRR,Add0R,Add0R]/2;

Iu = [elem(TL,:),NO+elem(TL,:),elem(TR,:),NO+elem(TR,:),2*NO+TL,2*NO+TR];
ii = [2*NO+repmat(TL,1,14),Iu,2*NO+repmat(TR,1,14),Iu];
jj = [Iu,2*NO+repmat(TL,1,14),Iu,2*NO+repmat(TR,1,14)];
ss = -mu_inEdge.*[valL,valL,valR,valR];

A = A + sparse(ii,jj,ss,DoF,DoF);

ii = [2*NO+TL,2*NO+TL,2*NO+TR,2*NO+TR];
jj = [2*NO+TL,2*NO+TR,2*NO+TL,2*NO+TR];
ss = mu_inEdge*rho1.*[dot(JumpL,JumpL,2),-dot(JumpL,JumpR,2),...
    -dot(JumpR,JumpL,2),dot(JumpR,JumpR,2)];
A = A + sparse(ii,jj,ss,DoF,DoF);
ss = rho1*[dot(JumpL,JumpL,2),-dot(JumpL,JumpR,2),...
    -dot(JumpR,JumpL,2),dot(JumpR,JumpR,2)];
B = B + sparse([TL,TL,TR,TR],[TL,TR,TL,TR],ss,NT,NT);

ii = [2*NO+TL,2*NO+TL,2*NO+TR,2*NO+TR,...
    2*NO+NT+TL,2*NO+NT+TR,2*NO+NT+TL,2*NO+NT+TR];
jj = [2*NO+NT+TL,2*NO+NT+TR,2*NO+NT+TL,2*NO+NT+TR...
    2*NO+TL,2*NO+TL,2*NO+TR,2*NO+TR];
ss = [Add0L,Add0L,Add0R,Add0R,Add0L,Add0L,Add0R,Add0R]/2;
A = A + sparse(ii,jj,ss,DoF,DoF);

%% Assemble Boundary Edge Terms

T.bdEdge = double(T.bdEdge);

xEbd = (node(T.bdEdge(:,1),:)+node(T.bdEdge(:,2),:))/2;
lengbdEdge = sqrt(sum((node(T.bdEdge(:,1),:)-node(T.bdEdge(:,2),:)).^2,2));
normVecbd = [(node(T.bdEdge(:,2),2)-node(T.bdEdge(:,1),2)),...
             (node(T.bdEdge(:,1),1)-node(T.bdEdge(:,2),1))];
normVecbd = normVecbd./lengbdEdge;

TB = T.bdEdge2elem;
mu_bd = mu_elem(TB);
DphiB = Dphi(TB,:,:);
JumpB = xEbd - xT(TB,:);

tmpL = dot(normVecbd,JumpB,2);
tmpIdx = tmpL<0;
normVecbd(tmpIdx,:) = -normVecbd(tmpIdx,:);

AcdBB = sum(DphiB.*normVecbd,2).*JumpB.*lengbdEdge;
Add0B = dot(normVecbd,JumpB,2).*lengbdEdge;

AcdBB = permute(AcdBB,[1 3 2]);
AcdBB = [AcdBB(:,:,1),AcdBB(:,:,2)];

Iu = [elem(TB,:),NO+elem(TB,:),2*NO+TB];
ii = [2*NO+repmat(TB,1,7),Iu];
jj = [Iu,2*NO+repmat(TB,1,7)];
ss = -mu_bd.*[AcdBB,Add0B,AcdBB,Add0B];
A = A + sparse(ii,jj,ss,DoF,DoF);

ii = 2*NO+TB;
jj = 2*NO+TB;
ss = mu_bd*rho1.*dot(JumpB,JumpB,2);
A = A + sparse(ii,jj,ss,DoF,DoF);

ii = [2*NO+TB,2*NO+NT+TB];
jj = [2*NO+NT+TB,2*NO+TB];
ss = [Add0B,Add0B];
A = A + sparse(ii,jj,ss,DoF,DoF);

%% Basis Functions (needed for the body-force reconstruction, TestType 2 & 3)

NiE = size(inEdge,1);

tmpL = double(T.edge2elem(inEdge,3));
edge2localnodeL = [mod(tmpL+1,3),mod(tmpL+2,3)];
edge2localnodeL = edge2localnodeL + 3*(edge2localnodeL == 0);
i1L = edge2localnodeL(:,1); i2L = edge2localnodeL(:,2);

tmpR = double(T.edge2elem(inEdge,4));
edge2localnodeR = [mod(tmpR+1,3),mod(tmpR+2,3)];
edge2localnodeR = edge2localnodeR + 3*(edge2localnodeR == 0);
i1R = edge2localnodeR(:,1); i2R = edge2localnodeR(:,2);

DphiL = [DphiL(:,:,1); DphiL(:,:,2); DphiL(:,:,3)];
DphiL1 = DphiL((1:NiE)'+(i1L-1)*NiE,:);
DphiL2 = DphiL((1:NiE)'+(i2L-1)*NiE,:);
DphiR = [DphiR(:,:,1); DphiR(:,:,2); DphiR(:,:,3)];
DphiR1 = DphiR((1:NiE)'+(i1R-1)*NiE,:);
DphiR2 = DphiR((1:NiE)'+(i2R-1)*NiE,:);

PhiDoFL = [-DphiL2(:,2),DphiL2(:,1)]-[-DphiL1(:,2),DphiL1(:,1)];
PhiDoFL = dot(PhiDoFL,normVecin,2).*(lenginEdge/2);
PhiDoFR = [-DphiR2(:,2),DphiR2(:,1)]-[-DphiR1(:,2),DphiR1(:,1)];
PhiDoFR = dot(PhiDoFR,normVecin,2).*(lenginEdge/2);

coefL = dot(JumpL,normVecin,2)/2.*lenginEdge;
coefR = dot(JumpR,normVecin,2)/2.*lenginEdge;

%% Assemble Velocity Mass Matrix M (for the Backward Euler (u,v)/dt term)

M = sparse(DoF_u,DoF_u);

for i = 1:3
    for j = 1:3
        % (v^C,w^C): always assembled directly, same for every TestType
        Mcc = localC(i,j)*(area/3);
        M = M + sparse([elem(:,i);NO+elem(:,i)],[elem(:,j);NO+elem(:,j)],[Mcc;Mcc],DoF_u,DoF_u);
    end

    if TestType ~= 3
        % (v^C,w^D) and (v^D,w^C), assembled directly (mass term unmodified)
        Mcd = zeros(NT,2);
        for p = 1:nQuad
            pxy = lambda(p,1)*node(elem(:,1),:) ...
                   + lambda(p,2)*node(elem(:,2),:) ...
                   + lambda(p,3)*node(elem(:,3),:);
            Mcd = Mcd + weight(p)*(pxy-xT)*lambda(p,i).*area;
        end
        ii = [elem(:,i),NO+elem(:,i),2*NO+TID,2*NO+TID];
        jj = [2*NO+TID,2*NO+TID,elem(:,i),NO+elem(:,i)];
        ss = [Mcd,Mcd];
        M = M + sparse(ii,jj,ss,DoF_u,DoF_u);
    end
end

if TestType ~= 3
    % (v^D,w^D), assembled directly (mass term unmodified)
    Mdd = zeros(NT,1);
    for p = 1:nQuad
        pxy = lambda(p,1)*node(elem(:,1),:) ...
                   + lambda(p,2)*node(elem(:,2),:) ...
                   + lambda(p,3)*node(elem(:,3),:);
        Mdd = Mdd + weight(p)*dot(pxy-xT,pxy-xT,2).*area;
    end
    M = M + sparse(2*NO+TID,2*NO+TID,Mdd,DoF_u,DoF_u);
else
    % (v^C,w^D) and (v^D,w^D), via the velocity reconstruction
    Dcdt1L = zeros(NiE,3); Dcdt2L = zeros(NiE,3);
    Dcdt1R = zeros(NiE,3); Dcdt2R = zeros(NiE,3);

    lengEdge = sqrt(sum((node(T.edge(:,1),:)-node(T.edge(:,2),:)).^2,2));
    normVec = [(node(T.edge(:,2),2)-node(T.edge(:,1),2)),...
                  (node(T.edge(:,1),1)-node(T.edge(:,2),1))];
    normVec = normVec./lengEdge;
    el2ed1 = T.elem2edge(:,1); el2ed2 = T.elem2edge(:,2); el2ed3 = T.elem2edge(:,3);
    xEmidT1 = xE(el2ed1,:); xEmidT2 = xE(el2ed2,:); xEmidT3 = xE(el2ed3,:);

    PhiDoF1 = [-Dphi(:,2,3), Dphi(:,1,3)] - [-Dphi(:,2,2), Dphi(:,1,2)];
    PhiDoF1 = dot(PhiDoF1,normVec(el2ed1,:),2).*(lengEdge(el2ed1)/2);
    PhiDoF2 = [-Dphi(:,2,3), Dphi(:,1,3)] - [-Dphi(:,2,1), Dphi(:,1,1)];
    PhiDoF2 = dot(PhiDoF2,normVec(el2ed2,:),2).*(lengEdge(el2ed2)/2);
    PhiDoF3 = [-Dphi(:,2,2), Dphi(:,1,2)] - [-Dphi(:,2,1), Dphi(:,1,1)];
    PhiDoF3 = dot(PhiDoF3,normVec(el2ed3,:),2).*(lengEdge(el2ed3)/2);

    coefT(:,1) = dot(xEmidT1-xT,normVec(el2ed1,:),2)/2.*lengEdge(el2ed1);
    coefT(:,2) = dot(xEmidT2-xT,normVec(el2ed2,:),2)/2.*lengEdge(el2ed2);
    coefT(:,3) = dot(xEmidT3-xT,normVec(el2ed3,:),2)/2.*lengEdge(el2ed3);

    ss = [coefT(:,1).*PhiDoF1, coefT(:,2).*PhiDoF2, coefT(:,3).*PhiDoF3];
    S = sparse(repmat(TID,1,3),double(T.elem2edge),ss,NT,NE);
    R = sparse(NE,NE);

    for p = 1:nQuad
        Phi(:,:,1) = phi(p,2)*[-Dphi(:,2,3), Dphi(:,1,3)] - phi(p,3)*[-Dphi(:,2,2), Dphi(:,1,2)];
        Phi(:,:,2) = phi(p,1)*[-Dphi(:,2,3), Dphi(:,1,3)] - phi(p,3)*[-Dphi(:,2,1), Dphi(:,1,1)];
        Phi(:,:,3) = phi(p,1)*[-Dphi(:,2,2), Dphi(:,1,2)] - phi(p,2)*[-Dphi(:,2,1), Dphi(:,1,1)];
        PhiL = phi(p,i1L)'.*[-DphiL2(:,2),DphiL2(:,1)] ...
                - phi(p,i2L)'.*[-DphiL1(:,2),DphiL1(:,1)];
        PhiR = phi(p,i1R)'.*[-DphiR2(:,2),DphiR2(:,1)] ...
                - phi(p,i2R)'.*[-DphiR1(:,2),DphiR1(:,1)];
        for i = 1:3
            Dcdt1L(:,i) = Dcdt1L(:,i) + weight(p)*phi(p,i)*PhiL(:,1).*area(TL);
            Dcdt2L(:,i) = Dcdt2L(:,i) + weight(p)*phi(p,i)*PhiL(:,2).*area(TL);
            Dcdt1R(:,i) = Dcdt1R(:,i) + weight(p)*phi(p,i)*PhiR(:,1).*area(TR);
            Dcdt2R(:,i) = Dcdt2R(:,i) + weight(p)*phi(p,i)*PhiR(:,2).*area(TR);
            for j = 1:3
                L2prod = weight(p)*dot(Phi(:,:,i),Phi(:,:,j),2).*area;
                R = R + sparse(T.elem2edge(:,i),T.elem2edge(:,j),L2prod,NE,NE);
            end
        end
    end

    Dcd1L = [coefL.*PhiDoFL.*Dcdt1L, coefL.*PhiDoFR.*Dcdt1R];
    Dcd2L = [coefL.*PhiDoFL.*Dcdt2L, coefL.*PhiDoFR.*Dcdt2R];
    Dcd1R = [coefR.*PhiDoFL.*Dcdt1L, coefR.*PhiDoFR.*Dcdt1R];
    Dcd2R = [coefR.*PhiDoFL.*Dcdt2L, coefR.*PhiDoFR.*Dcdt2R];

    ii = [2*NO+repmat(TL,1,6),2*NO+repmat(TL,1,6),2*NO+repmat(TR,1,6),2*NO+repmat(TR,1,6)];
    jj = [elem(TL,:),elem(TR,:),NO+elem(TL,:),NO+elem(TR,:), ...
        elem(TL,:),elem(TR,:),NO+elem(TL,:),NO+elem(TR,:)];
    ss = [Dcd1L,Dcd2L,Dcd1R,Dcd2R];
    M = M + sparse(ii,jj,ss,DoF_u,DoF_u) + sparse(jj,ii,ss,DoF_u,DoF_u);

    R(bdEdge,:) = R(bdEdge,:)*0; R(:,bdEdge) = R(:,bdEdge)*0;
    R = S*R*S';
    M(2*NO+1:2*NO+NT,2*NO+1:2*NO+NT) = M(2*NO+1:2*NO+NT,2*NO+1:2*NO+NT) + R;
end

A(1:DoF_u,1:DoF_u) = A(1:DoF_u,1:DoF_u) + M/dt;

%% Boundary Condition Setup

bdNode = unique([T.bdEdge(:,1);T.bdEdge(:,2)]);
isBdDoF = false(DoF,1);
isBdDoF([bdNode;NO+bdNode;end]) = true;
freeDoF = find(~isBdDoF);

%% Initialize State at t=0

t = 0;
u0 = pde.exact_u(node,0);
u0 = [u0(:); zeros(NT,1)];

sum_axp2 = 0;
sum_u_E2 = 0;

%% Time Loop

for it = 1:time_it
    t = t + dt;

    F = zeros(DoF,1);
    ft1 = zeros(NT,3); ft2 = zeros(NT,3); f0 = zeros(NT,1); pt = zeros(NT,1);

    if TestType == 1
        for p = 1:nQuad
            pxy = lambda(p,1)*node(elem(:,1),:) ...
                + lambda(p,2)*node(elem(:,2),:) ...
                + lambda(p,3)*node(elem(:,3),:);
            fp = pde.rhs(pxy,t);
            pp = pde.exact_p(pxy,t);
            for j = 1:3
                ft1(:,j) = ft1(:,j) + weight(p)*phi(p,j)*fp(:,1);
                ft2(:,j) = ft2(:,j) + weight(p)*phi(p,j)*fp(:,2);
            end
            f0 = f0 + weight(p)*dot(pxy-xT,fp,2).*area;
            pt = pt + weight(p)*pp;
        end

        ft1 = ft1.*repmat(area,1,3);
        ft2 = ft2.*repmat(area,1,3);

        f1 = accumarray(elem(:),ft1(:),[NO 1]);
        f2 = accumarray(elem(:),ft2(:),[NO 1]);

        F(1:2*NO+NT) = [f1;f2;f0];
    else % TestType == 2 or 3: pressure-robust body force via reconstruction
        ftL = zeros(NiE,1); ftR = zeros(NiE,1);

        for p = 1:nQuad
            pxy = lambda(p,1)*node(elem(:,1),:) ...
                + lambda(p,2)*node(elem(:,2),:) ...
                + lambda(p,3)*node(elem(:,3),:);
            pxyL = pxy(TL,:); pxyR = pxy(TR,:);
            fp = pde.rhs(pxy,t); fpL = pde.rhs(pxyL,t); fpR = pde.rhs(pxyR,t);
            pp = pde.exact_p(pxy,t);
            PhiL = phi(p,i1L)'.*[-DphiL2(:,2),DphiL2(:,1)] ...
                    - phi(p,i2L)'.*[-DphiL1(:,2),DphiL1(:,1)];
            PhiR = phi(p,i1R)'.*[-DphiR2(:,2),DphiR2(:,1)] ...
                    - phi(p,i2R)'.*[-DphiR1(:,2),DphiR1(:,1)];
            ftL = ftL + weight(p)*dot(fpL,PhiL,2).*area(TL);
            ftR = ftR + weight(p)*dot(fpR,PhiR,2).*area(TR);
            for j = 1:3
                ft1(:,j) = ft1(:,j) + weight(p)*phi(p,j)*fp(:,1);
                ft2(:,j) = ft2(:,j) + weight(p)*phi(p,j)*fp(:,2);
            end
            pt = pt + weight(p)*pp;
        end

        fL = coefL.*ftL.*PhiDoFL + coefL.*ftR.*PhiDoFR;
        fR = coefR.*ftR.*PhiDoFR + coefR.*ftL.*PhiDoFL;
        fL = accumarray(TL,fL,[NT 1]);
        fR = accumarray(TR,fR,[NT 1]);
        ft1 = ft1.*repmat(area,1,3);
        ft2 = ft2.*repmat(area,1,3);
        f1 = accumarray(elem(:),ft1(:),[NO 1]);
        f2 = accumarray(elem(:),ft2(:),[NO 1]);

        F(1:2*NO+NT) = [f1;f2;(fL+fR)];
    end

    F(1:2*NO+NT) = F(1:2*NO+NT) + M*u0/dt;

    %% Solve Linear System

    x = zeros(DoF,1);
    uBd = pde.exact_u(node,t);
    x(bdNode) = uBd(bdNode,1);
    x(NO+bdNode) = uBd(bdNode,2);
    x(end) = pt(end);

    F = F - A(:,isBdDoF)*x(isBdDoF);

    x(freeDoF) = A(freeDoF,freeDoF)\F(freeDoF);

    u0 = x(1:2*NO+NT);

    %% Compute Errors

    ph = x(2*NO+NT+1:2*NO+2*NT);

    uhCp = zeros(NO,2); uhCp(:,1) = x(1:NO); uhCp(:,2) = x(NO+1:2*NO);
    uhD = x(2*NO+1:2*NO+NT);
    Duh1 = Dphi(:,:,1).*uhCp(elem(:,1),1) + Dphi(:,:,2).*uhCp(elem(:,2),1)...
            + Dphi(:,:,3).*uhCp(elem(:,3),1) + uhD*[1 0];
    Duh2 = Dphi(:,:,1).*uhCp(elem(:,1),2) + Dphi(:,:,2).*uhCp(elem(:,2),2)...
            + Dphi(:,:,3).*uhCp(elem(:,3),2) + uhD*[0 1];

    err_Dut = zeros(NT,1); err_ut = zeros(NT,1); err_pt = zeros(NT,1);
    for p = 1:nQuad
        pxy = lambda(p,1)*node(elem(:,1),:) ...
                + lambda(p,2)*node(elem(:,2),:) ...
                + lambda(p,3)*node(elem(:,3),:);
        uh = phi(p,1)*uhCp(elem(:,1),:) + phi(p,2)*uhCp(elem(:,2),:) ...
                + phi(p,3)*uhCp(elem(:,3),:) + uhD.*(pxy-xT);
        Du = pde.exact_gu(pxy,t); uExa = pde.exact_u(pxy,t); pval = pde.exact_p(pxy,t);
        err_Dut = err_Dut + weight(p)*(sum((Duh1-Du(:,:,1)).^2,2)...
                                + sum((Duh2-Du(:,:,2)).^2,2));
        err_ut = err_ut + weight(p)*sum((uExa-uh).^2,2);
        err_pt = err_pt + weight(p)*(pval-ph).^2;
    end
    err_Dut = err_Dut.*area; err_ut = err_ut.*area; err_pt = err_pt.*area;

    err_u0 = sqrt(sum(err_ut));
    err_u = sqrt(sum(err_Dut) + (uhD'*B*uhD));
    err_axp = sqrt(sum((ph-pt).^2.*area));
    err_p = sqrt(sum(err_pt));
    sum_axp2 = sum_axp2 + err_axp^2;
    sum_u_E2 = sum_u_E2 + err_u^2;

end

err_axp_time = sqrt(dt * sum_axp2);
err_u_time = sqrt(dt * sum_u_E2);

fprintf('\n    ||u-u_h||_L2E : %.3e    ||u-u_h||_0 : %.3e    ||P_0p-p_h||_0 : %.3e    ||p-p_h||_0 : %.3e    ||P_0p-p_h||_L2L2 : %.3e\n\n',err_u_time,err_u0,err_axp,err_p,err_axp_time)
