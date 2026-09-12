%% MAIN_EGBR2: Enriched Galerkin Method for the Brinkman Equations (2D)
%
% MAIN_EGBR2 produces a numerical solution using the Enriched Galerkin
% (EG) method for the Brinkman equations in a 2-dimensional domain:
%
%   - div(nu*grad(u)) + K*u + grad(p) = f in Omega
%                            - div(u) = 0 in Omega
%
% with a Dirichlet boundary condition. Continuous piecewise linear
% functions and discontinuous piecewise constant functions are used for
% the velocity and the pressure, respectively, with an additional
% discontinuous linear enrichment (x - x_T per element T) added to the
% velocity.
%
% TestType controls how the body force f and the reaction term K*(u,v)
% (the "mass term") are assembled:
%   TestType = 1: standard EG. Both the body force and the K*(u,v)
%                 reaction term are assembled directly, elementwise.
%   TestType = 2: pressure-robust body force only. The body force is
%                 assembled through the velocity reconstruction
%                 operator, while the K*(u,v) reaction term is left
%                 unmodified, exactly as in TestType 1.
%   TestType = 3: pressure-robust body force AND reaction term. Both the
%                 body force and the K*(u,v) reaction term are assembled
%                 through the velocity reconstruction operator.
%                 Changing the body force alone (TestType 2) is not
%                 enough to obtain pressure-robust results for the
%                 Brinkman equations; the reaction/mass term must be
%                 reconstructed the same way, which is what TestType 3
%                 does.
%
% Necessary m-files from iFEM (by L. Chen)
%   auxstructure.m
%   squaremesh.m
%   gradbasis.m
% included in iFEM_files folder.
%
% See also: Brinkman2.m, as an example.
%
% References:
%   'An enriched Galerkin method for the Stokes equations' by
%     S.-Y. Yi, X. Hu, S. Lee, and J. H. Adler, 2022.
%   'Pressure-robust enriched Galerkin methods for the Stokes equations'
%     by X. Hu, S. Lee, L. Mu, and S.-Y. Yi, 2024.
%   'A Uniform and Pressure-Robust Enriched Galerkin Method for the
%     Brinkman Equations' by S. Lee and L. Mu, 2024.
%
% Author: Seulip Lee and Lin Mu
%
clear
close all

%% Preliminaries

Num = 16;
rho1 = 3; rho2 = 0;
TestType = 3; % 1: standard / 2: PR body force only / 3: PR body force + reaction term
assert(ismember(TestType,[1 2 3]), 'TestType must be 1, 2, or 3.');
pde = Brinkman2; nu_fun = pde.nu; K_fun = pde.K;

% Mesh generation
[node,elem] = squaremesh([0,1,0,1],1/Num);
T = auxstructure(elem);
[Dphi,area] = gradbasis(node,elem);
xT = (node(elem(:,1),:)+node(elem(:,2),:)+node(elem(:,3),:))/3;
xE = (node(T.edge(:,1),:)+node(T.edge(:,2),:))/2;

% Number of elements, edges, nodes
NT = size(elem,1); NE = size(T.edge,1); NO = size(node,1);

% Degrees of freedom
DoF_u = (2*NO+NT); DoF_p = NT; DoF = DoF_u + DoF_p;

% K and nu are defined at each element's centroid (not averaged from
% vertices), so a sharp permeability/viscosity interface between elements
% stays sharp instead of being blended across the boundary element.
K_elem = K_fun(xT);
nu_elem = nu_fun(xT);

[lambda,weight] = quadpts(9); phi = lambda; nQuad = size(lambda,1);

%% Assemble Element Terms

A = sparse(DoF,DoF);
TID = (1:NT)';
localC = [2 1 1; 1 2 1; 1 1 2]/4; % from quadrature rule

for i = 1:3
    for j = 1:3
        % nu*(grad u^C, grad v^C)
        Acc = nu_elem.*dot(Dphi(:,:,i),Dphi(:,:,j),2).*area;
        % K*(u^C,v^C) -- assembled directly for every TestType
        Ccc = localC(i,j)*K_elem.*(area/3); % higher order needed
        A = A + sparse([elem(:,i);NO+elem(:,i)],[elem(:,j);NO+elem(:,j)],[Acc+Ccc;Acc+Ccc],DoF,DoF);
    end
    Acp = Dphi(:,:,i).*area; % -(div u^C, q) and -(div v^C, p)
    Acd = nu_elem.*Acp; %nu*(grad u^C, grad v^D) and nu*(grad u^D, grad v^C)

    if TestType == 3
        Ccd = Acd*0; % reaction cross term is reconstructed below instead
    else
        % K*(u^C,v^D) and K*(u^D,v^C), assembled directly (mass term unmodified)
        Ccd = zeros(NT,2);
        for p = 1:nQuad
            pxy = lambda(p,1)*node(elem(:,1),:) ...
                   + lambda(p,2)*node(elem(:,2),:) ...
                   + lambda(p,3)*node(elem(:,3),:);
            Ccd = Ccd + weight(p)*(pxy-xT)*lambda(p,i).*area.*K_elem;
        end
    end

    ii = [elem(:,i),NO+elem(:,i),2*NO+TID,2*NO+TID,...
        elem(:,i),NO+elem(:,i),2*NO+NT+TID,2*NO+NT+TID];
    jj = [2*NO+TID,2*NO+TID,elem(:,i),NO+elem(:,i),...
        2*NO+NT+TID,2*NO+NT+TID,elem(:,i),NO+elem(:,i)];
    ss = [Acd+Ccd,Acd+Ccd,-Acp,-Acp];
    A = A + sparse(ii,jj,ss,DoF,DoF);
end

Adp = 2*area; % -(div v^D, p) and -(div u^D, q)
Add = nu_elem.*Adp;

if TestType == 3
    Cdd = Add*0; % reaction D-D term is reconstructed below instead
else
    % K*(u^D,v^D), assembled directly (mass term unmodified)
    Cdd = zeros(NT,1);
    for p = 1:nQuad
        pxy = lambda(p,1)*node(elem(:,1),:) ...
                   + lambda(p,2)*node(elem(:,2),:) ...
                   + lambda(p,3)*node(elem(:,3),:);
        Cdd = Cdd + weight(p)*dot(pxy-xT,pxy-xT,2).*area.*K_elem;
    end
end
A = A + sparse([2*NO+TID,2*NO+TID,2*NO+NT+TID], ...
    [2*NO+TID,2*NO+NT+TID,2*NO+TID],[Add+Cdd,-Adp,-Adp],DoF,DoF);

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

% K is the harmonic mean of the two adjacent elements' values (the
% standard interface-permeability average for porous media), while nu
% (a diffusion/DG-penalty coefficient, not a flux-continuity coefficient)
% uses the usual arithmetic average.
K_inEdge = 2*K_elem(TL).*K_elem(TR)./(K_elem(TL)+K_elem(TR));
nu_inEdge = (nu_elem(TL)+nu_elem(TR))/2;

tmpL = dot(normVecin,xT(TR,:)-xT(TL,:),2);
tmpIdx = tmpL<0;
normVecin(tmpIdx,:) = -normVecin(tmpIdx,:);

DphiL = Dphi(TL,:,:); DphiR = Dphi(TR,:,:);
JumpL = xEin - xT(TL,:); JumpR = xEin - xT(TR,:);

% nu*<{grad u}n, [v^D]>
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
ss = -nu_inEdge.*[valL,valL,valR,valR];

A = A + sparse(ii,jj,ss,DoF,DoF);

% nu*rho1/h*<[u^D],[v^D]> (+ rho2*K jump, if enabled)
ii = [2*NO+TL,2*NO+TL,2*NO+TR,2*NO+TR];
jj = [2*NO+TL,2*NO+TR,2*NO+TL,2*NO+TR];
ss = (nu_inEdge*rho1+rho2*K_inEdge.*lenginEdge.^2).*[dot(JumpL,JumpL,2),-dot(JumpL,JumpR,2),...
    -dot(JumpR,JumpL,2),dot(JumpR,JumpR,2)];
A = A + sparse(ii,jj,ss,DoF,DoF);
ss = rho1*[dot(JumpL,JumpL,2),-dot(JumpL,JumpR,2),...
    -dot(JumpR,JumpL,2),dot(JumpR,JumpR,2)];
B = B + sparse([TL,TL,TR,TR],[TL,TR,TL,TR],ss,NT,NT);

% <{p},[v^D]> & <{q},[u^D]>
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
% Only one element borders a boundary edge, so its value is used directly.
K_bd = K_elem(TB);
nu_bd = nu_elem(TB);
DphiB = Dphi(TB,:,:);
JumpB = xEbd - xT(TB,:);

tmpL = dot(normVecbd,JumpB,2);
tmpIdx = tmpL<0;
normVecbd(tmpIdx,:) = -normVecbd(tmpIdx,:);

% nu*<{grad u}n,[v^D]>
AcdBB = sum(DphiB.*normVecbd,2).*JumpB.*lengbdEdge;
Add0B = dot(normVecbd,JumpB,2).*lengbdEdge;

AcdBB = permute(AcdBB,[1 3 2]);
AcdBB = [AcdBB(:,:,1),AcdBB(:,:,2)];

Iu = [elem(TB,:),NO+elem(TB,:),2*NO+TB];
ii = [2*NO+repmat(TB,1,7),Iu];
jj = [Iu,2*NO+repmat(TB,1,7)];
ss = -nu_bd.*[AcdBB,Add0B,AcdBB,Add0B];
A = A + sparse(ii,jj,ss,DoF,DoF);

% nu*rho1/h*<[u^D],[v^D]> (+ rho2*K jump, if enabled)
ii = 2*NO+TB;
jj = 2*NO+TB;
ss = (nu_bd*rho1+rho2*K_bd.*lengbdEdge.^2).*dot(JumpB,JumpB,2);
A = A + sparse(ii,jj,ss,DoF,DoF);
ss = rho1.*dot(JumpB,JumpB,2);
B = B + sparse(TB,TB,ss,NT,NT);

% <{p},[v^D]>
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

%% Assemble Reaction-Term (Mass-Term) Reconstruction -- TestType 3 only
%
% Same edge-based, velocity reconstruction applied here to the K*(u,v)
% reaction term. K_elem/K_inEdge weights every piece so this stays
% correct once K is spatially varying (e.g. a permeability map).

if TestType == 3
    D = sparse(DoF_u,DoF_u);

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
                % K*(reconstructed basis_i, reconstructed basis_j) over each element
                L2prod = weight(p)*dot(Phi(:,:,i),Phi(:,:,j),2).*area.*K_elem;
                R = R + sparse(T.elem2edge(:,i),T.elem2edge(:,j),L2prod,NE,NE);
            end
        end
    end

    % K*(u^C,v^D) and K*(u^D,v^C), via the reconstruction operator
    Dcd1L = [coefL.*PhiDoFL.*Dcdt1L, coefL.*PhiDoFR.*Dcdt1R];
    Dcd2L = [coefL.*PhiDoFL.*Dcdt2L, coefL.*PhiDoFR.*Dcdt2R];
    Dcd1R = [coefR.*PhiDoFL.*Dcdt1L, coefR.*PhiDoFR.*Dcdt1R];
    Dcd2R = [coefR.*PhiDoFL.*Dcdt2L, coefR.*PhiDoFR.*Dcdt2R];

    ii = [2*NO+repmat(TL,1,6),2*NO+repmat(TL,1,6),2*NO+repmat(TR,1,6),2*NO+repmat(TR,1,6)];
    jj = [elem(TL,:),elem(TR,:),NO+elem(TL,:),NO+elem(TR,:), ...
        elem(TL,:),elem(TR,:),NO+elem(TL,:),NO+elem(TR,:)];
    ss = [Dcd1L,Dcd2L,Dcd1R,Dcd2R];
    D = D + sparse(ii,jj,K_inEdge.*ss,DoF_u,DoF_u) + sparse(jj,ii,K_inEdge.*ss,DoF_u,DoF_u);

    % K*(u^D,v^D), via the reconstruction operator
    R(bdEdge,:) = R(bdEdge,:)*0; R(:,bdEdge) = R(:,bdEdge)*0;
    R = S*R*S';
    D(2*NO+1:2*NO+NT,2*NO+1:2*NO+NT) = D(2*NO+1:2*NO+NT,2*NO+1:2*NO+NT) + R;

    % Energy-error matrix: Stokes part of the D-D block only (reaction
    % term excluded), matching the convention used for TestType 1 and 2.
    % Assumes nu is constant over the mesh (as in the reference
    % time-dependent code); replace nu_elem(1) with a proper nu-weighted
    % quadrature if nu becomes spatially varying.
    B = A(2*NO+1:2*NO+NT,2*NO+1:2*NO+NT)/nu_elem(1);

    A(1:DoF_u,1:DoF_u) = A(1:DoF_u,1:DoF_u) + D;
end

%% Assemble Right Hand Side

F = zeros(DoF,1);
ft1 = zeros(NT,3); ft2 = zeros(NT,3); f0 = zeros(NT,1); pt = zeros(NT,1);

if TestType == 1
    for p = 1:nQuad
        pxy = lambda(p,1)*node(elem(:,1),:) ...
            + lambda(p,2)*node(elem(:,2),:) ...
            + lambda(p,3)*node(elem(:,3),:);
        fp = pde.rhs(pxy);
        pp = pde.exact_p(pxy);
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
        fp = pde.rhs(pxy); fpL = pde.rhs(pxyL); fpR = pde.rhs(pxyR);
        pp = pde.exact_p(pxy);
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

%% Solve Linear System

x = zeros(DoF,1);

bdNode = unique([T.bdEdge(:,1);T.bdEdge(:,2)]);
isBdDoF = false(DoF,1);
isBdDoF([bdNode;NO+bdNode;end]) = true;
freeDoF = find(~isBdDoF);

uBd = pde.exact_u(node);
x(bdNode) = uBd(bdNode,1);
x(NO+bdNode) = uBd(bdNode,2);
x(end) = pt(end);

F = F - A(:,isBdDoF)*x(isBdDoF);

x(freeDoF) = A(freeDoF,freeDoF)\F(freeDoF);

Nu_free = sum(~isBdDoF(1:DoF_u));
Np_free = DoF_p - 1; % use the fact that the last pressure dof is fixed
Mp = spdiags(area(1:end-1), 0, Np_free, Np_free); % use the fact that the last pressure dof is fixed

% Optional solvers (need Mp, Nu_free, Np_free built above):
%   x(freeDoF) = exact_block_precond_solver(A(freeDoF,freeDoF), F(freeDoF), Nu_free, Np_free, Mp, nu_fun(0));
%   x(freeDoF) = inexact_block_precond_solver(A(freeDoF,freeDoF), F(freeDoF), Nu_free, Np_free, Mp, nu_fun(0));

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
    Du = pde.exact_gu(pxy); uExa = pde.exact_u(pxy); pval = pde.exact_p(pxy);
    err_Dut = err_Dut + weight(p)*(sum((Duh1-Du(:,:,1)).^2,2)...
                            + sum((Duh2-Du(:,:,2)).^2,2));
    err_ut = err_ut + weight(p)*sum((uExa-uh).^2,2);
    err_pt = err_pt + weight(p)*(pval-ph).^2;
end
err_Dut = err_Dut.*area; err_ut = err_ut.*area; err_pt = err_pt.*area;

err_u0 = sqrt(sum(err_ut));
% Discrete H1-type seminorm ||u-u_h||_E, NOT scaled by nu (nu_elem multiplies
% neither err_Dut nor B here).
err_u = sqrt(sum(err_Dut) + (uhD'*B*uhD));
% Brinkman energy norm |||u-u_h|||^2 = nu*||u-u_h||_E^2 + ||u-u_h||_0^2, with nu
% applied explicitly at this combination step (nu_elem.*uhD lets this stay
% correct once nu/K become spatially varying, e.g. a permeability map).
err_triple = sqrt(sum(nu_elem.*err_Dut) + uhD'*B*(nu_elem.*uhD) + sum(err_ut));
err_axp = sqrt(sum((ph-pt).^2.*area));
err_p = sqrt(sum(err_pt));

fprintf('\n    ||u-u_h||_E : %f    ||u-u_h||_0 : %f    |||u-u_h||| : %f    ||P_0p-p_h||_0 : %f    ||p-p_h||_0 : %f\n\n',err_u,err_u0,err_triple,err_axp,err_p)

%% Plot Numerical Solutions

uhp = zeros(NO,1); vhp = zeros(NO,1); count = zeros(NO,1);
uhDp = zeros(3,NT); vhDp = zeros(3,NT);
for i = 1:NT
    uhp(elem(i,:)) = uhp(elem(i,:)) + x(2*NO+i)*(node(elem(i,:),1)-xT(i,1));
    vhp(elem(i,:)) = vhp(elem(i,:)) + x(2*NO+i)*(node(elem(i,:),2)-xT(i,2));
    count(elem(i,:)) = count(elem(i,:)) + 1;

    uhDp(:,i) = x(2*NO+i)*(node(elem(i,:),1)-xT(i,1));
    vhDp(:,i) = x(2*NO+i)*(node(elem(i,:),2)-xT(i,2));
end
uhp = x(1:NO) + uhp./count;
vhp = x(NO+1:2*NO) + vhp./count;

uhDp = uhDp(:); vhDp = vhDp(:);

elem_tr = elem'; P = node(elem_tr(:),:);
uhCp = x(elem_tr(:),:); vhCp = x(elem_tr(:)+NO,:);
tri=reshape(1:size(P,1),[3 size(P,1)/3]);

% Velocity components
figure
colormap jet
trisurf(tri',P(:,1),P(:,2),uhCp+uhDp,'EdgeColor', 'none');
view(2); axis square;
ax = gca; ax.FontSize = 20;
box on
c = colorbar; c.FontSize = 20; c.Location = 'southoutside';

figure
colormap jet
trisurf(tri',P(:,1),P(:,2),vhCp+vhDp,'EdgeColor', 'none');
view(2); axis square;
ax = gca; ax.FontSize = 20;
box on
c = colorbar; c.FontSize = 20; c.Location = 'southoutside';

% Velocity magnitude
figure
colormap jet
trisurf(tri',P(:,1),P(:,2),sqrt((uhCp+uhDp).^2+(vhCp+vhDp).^2),'EdgeColor', 'none');
view(2); axis square;
ax = gca; ax.FontSize = 20;
box on
c = colorbar; c.FontSize = 20; c.Location = 'southoutside';

% Pressure
figure
colormap jet
trisurf(tri',P(:,1),P(:,2),[ph';ph';ph'],'EdgeColor', 'none');
view(2); axis square; axis([0,1,0,1]);
c = colorbar; c.FontSize = 20; c.Location = 'southoutside';
ax = gca; ax.SortMethod = "childorder"; ax.FontSize = 20;
box on
