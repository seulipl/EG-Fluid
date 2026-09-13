%% MAIN_EG3: Enriched Galerkin Method for Stokes Equations
% 
% MAIN_EG3 produces approximate solution using a pressure-robust Enriched
% Galerkin (EG) method for the Stokes equations in 3-dimensional spaces:
% 
%   - div(nu*grad(u)) + grad(p) = f in Omega
%                      - div(u) = 0 in Omega
% 
% with a Dirichlet boundary condition. In the EG method, continuous
% piecewise linear functions and discontinuous piecewise constant functions
% are applied for the velocity and the pressure, respectively. Moreover,
% additional degrees of freedom are added to the velocity, and they are
% defined as discontinuous linear basis functions, that is, x - x_T for
% each element T.
% 
% For pressure robust scheme, a divergence-preserving velocity
% reconstruction operator is applied to modify the body force assembling on
% the right hand side, so the same stiffness matrix will be used. The
% variable 'TestType' indicates the standard EG method (TestType = 1) or
% the pressure-robust EG method (TestType = 2).
%
% Dependencies (from iFEM by L. Chen):
%   auxstructure3.m, cubemesh.m, gradbasis3.m, mycross.m, myunique.m, quadpts3.m
%
% See also: Stokes3.m, as examples.
%
% References:
%   X. Hu, S. Lee, L. Mu, and S.-Y. Yi, "Pressure-robust enriched Galerkin
%     methods for the Stokes equations," Journal of Computational and
%     Applied Mathematics, 2024.
%
% Authors: Seulip Lee and Lin Mu
%
clear

%% Preliminaries

% Parameters
Deg = 1;
Num = 16; % 1/Num = h
pde = Stokes3;
rho = 3; % penalty parameter
TestType = 2;
nu_val = pde.nu; % fluid viscosity

% Mesh generation
[node,elem] = cubemesh([0,1,0,1,0,1],1/Num);
T = auxstructure3(elem);
[Dphi,volume] = gradbasis3(node,elem);

% Number of elements, faces, nodes
NT = size(elem,1);
NF = size(T.face,1);
NO = size(node,1);

% Degrees of freedom
DoF_u = (3*NO+NT);
DoF_p = NT;
DoF = DoF_u + DoF_p;

%% Assemble Element Terms

A = sparse(DoF,DoF);
B1= sparse(3*NO,3*NO); % matrix for checking energy error in u^C
TID = (1:NT)';
for i = 1:4
    for j = 1:4
        % nu*(grad u^C, grad v^C)
        Acc = nu_val*dot(Dphi(:,:,i),Dphi(:,:,j),2).*volume; 
        A = A + sparse([elem(:,i);NO+elem(:,i);2*NO+elem(:,i)],...
            [elem(:,j);NO+elem(:,j);2*NO+elem(:,j)],[Acc;Acc;Acc],DoF,DoF);
        B1 = B1 + sparse([elem(:,i);NO+elem(:,i);2*NO+elem(:,i)],...
            [elem(:,j);NO+elem(:,j);2*NO+elem(:,j)],[Acc;Acc;Acc],3*NO,3*NO);
    end
    Acp = Dphi(:,:,i).*volume; % -(div u^C, q) and -(div v^C, p)
    Acd = nu_val*Acp;  %nu*(grad u^C, grad v^D) and nu*(grad u^D, grad v^C)
    ii = [elem(:,i),NO+elem(:,i),2*NO+elem(:,i),...
        3*NO+TID,3*NO+TID,3*NO+TID,...
        elem(:,i),NO+elem(:,i),2*NO+elem(:,i),...
        3*NO+NT+TID,3*NO+NT+TID,3*NO+NT+TID];
    jj = [3*NO+TID,3*NO+TID,3*NO+TID,...
        elem(:,i),NO+elem(:,i),2*NO+elem(:,i),...
        3*NO+NT+TID,3*NO+NT+TID,3*NO+NT+TID,...
        elem(:,i),NO+elem(:,i),2*NO+elem(:,i)];
    ss = [Acd,Acd,-Acp,-Acp];
    A = A + sparse(ii,jj,ss,DoF,DoF);
end

Adp = 3*volume; % -(div v^D, p) and -(div u^D, q)
Add = nu_val*Adp; % nu*(grad u^D, grad v^D)
A = A + sparse([3*NO+TID,3*NO+TID,3*NO+NT+TID], ...
    [3*NO+TID,3*NO+NT+TID,3*NO+TID],[Add,-Adp,-Adp],DoF,DoF);

%% Assemble Interior Face Terms

B2 = sparse(NT,NT); % matrix for checking energy error in u^D

bdFace = find(T.face2elem(:,1)==T.face2elem(:,2));
inFace = setdiff((1:NF)',bdFace);

xF = (node(T.face(inFace,1),:)+node(T.face(inFace,2),:)+node(T.face(inFace,3),:))/3;
xT = (node(elem(:,1),:)+node(elem(:,2),:)+node(elem(:,3),:)+node(elem(:,4),:))/4;

r12 = node(T.face(inFace,2),:) - node(T.face(inFace,1),:);
r13 = node(T.face(inFace,3),:) - node(T.face(inFace,1),:);
normVecin = cross(r12,r13,2);
areainFace = vecnorm(normVecin,2,2)/2;
normVecin = normVecin./(areainFace*2);

TL = double(T.face2elem(inFace,1));
TR = double(T.face2elem(inFace,2));

tmpL = dot(normVecin,xT(TR,:)-xT(TL,:),2);
tmpIdx = tmpL<0; % normal vector should be from left to right cell
normVecin(tmpIdx,:) = -normVecin(tmpIdx,:);

DphiL = Dphi(TL,:,:); DphiR = Dphi(TR,:,:);
JumpL = xF - xT(TL,:); JumpR = xF - xT(TR,:);

% nu*<{grad u}n, [v^D]>
AcdLL = sum(DphiL.*normVecin,2).*JumpL.*areainFace; % nu*<{grad u^C}n, [v^D]>
AcdRL = sum(DphiR.*normVecin,2).*JumpL.*areainFace;
Add0L = dot(normVecin,JumpL,2).*areainFace; % nu*<{grad u^D}n, [v^D]>

AcdRR = -sum(DphiR.*normVecin,2).*JumpR.*areainFace;
AcdLR = -sum(DphiL.*normVecin,2).*JumpR.*areainFace;
Add0R = -dot(normVecin,JumpR,2).*areainFace;

AcdLL = permute(AcdLL,[1 3 2]);
AcdRL = permute(AcdRL,[1 3 2]);
AcdRR = permute(AcdRR,[1 3 2]);
AcdLR = permute(AcdLR,[1 3 2]);

AcdLL = [AcdLL(:,:,1),AcdLL(:,:,2),AcdLL(:,:,3)];
AcdRL = [AcdRL(:,:,1),AcdRL(:,:,2),AcdRL(:,:,3)];
AcdLR = [AcdLR(:,:,1),AcdLR(:,:,2),AcdLR(:,:,3)];
AcdRR = [AcdRR(:,:,1),AcdRR(:,:,2),AcdRR(:,:,3)];

valL = [AcdLL,AcdRL,Add0L,Add0L]/2;
valR = [AcdLR,AcdRR,Add0R,Add0R]/2;

Iu = [elem(TL,:),NO+elem(TL,:),2*NO+elem(TL,:),...
    elem(TR,:),NO+elem(TR,:),2*NO+elem(TR,:),...
    3*NO+TL,3*NO+TR];

ii = 3*NO+repmat(TL,1,26);
jj = Iu;
ss = -nu_val*valL;
A = A + sparse(ii,jj,ss,DoF,DoF);
ii = Iu;
jj = 3*NO+repmat(TL,1,26);
ss = -nu_val*valL;
A = A + sparse(ii,jj,ss,DoF,DoF);
ii = 3*NO+repmat(TR,1,26);
jj = Iu;
ss = -nu_val*valR;
A = A + sparse(ii,jj,ss,DoF,DoF);
ii = Iu;
jj = 3*NO+repmat(TR,1,26);
ss = -nu_val*valR;
A = A + sparse(ii,jj,ss,DoF,DoF);

% nu*rho/h*<[u^D],[v^D]>
ii = [3*NO+TL,3*NO+TL,3*NO+TR,3*NO+TR];
jj = [3*NO+TL,3*NO+TR,3*NO+TL,3*NO+TR];
ss = nu_val*rho*[dot(JumpL,JumpL,2),-dot(JumpL,JumpR,2),...
    -dot(JumpR,JumpL,2),dot(JumpR,JumpR,2)].*sqrt(areainFace);
    
A = A + sparse(ii,jj,ss,DoF,DoF);
B2 = B2 + sparse([TL,TL,TR,TR],[TL,TR,TL,TR],ss,NT,NT);

% <{p},[v^D]> & <{q},[u^D]>

ii = [3*NO+TL,3*NO+TL,3*NO+TR,3*NO+TR,...
    3*NO+NT+TL,3*NO+NT+TR,3*NO+NT+TL,3*NO+NT+TR];
jj = [3*NO+NT+TL,3*NO+NT+TR,3*NO+NT+TL,3*NO+NT+TR,...
    3*NO+TL,3*NO+TL,3*NO+TR,3*NO+TR];
ss = [Add0L,Add0L,Add0R,Add0R,Add0L,Add0L,Add0R,Add0R]/2;
A = A + sparse(ii,jj,ss,DoF,DoF);

Ain = A;

%% Assemble Boundary Face Terms

T.bdFace = double(T.bdFace);

xFbd = (node(T.bdFace(:,1),:)+node(T.bdFace(:,2),:)+node(T.bdFace(:,3),:))/3;
r12 = node(T.bdFace(:,2),:) - node(T.bdFace(:,1),:);
r13 = node(T.bdFace(:,3),:) - node(T.bdFace(:,1),:);
normVecbd = cross(r13,r12,2);
areabdFace = vecnorm(normVecbd,2,2)/2;
normVecbd = normVecbd./(areabdFace*2);

TB = T.bdFace2elem;
DphiB = Dphi(TB,:,:);
JumpB = xFbd - xT(TB,:);

tmpL = dot(normVecbd,JumpB,2);
tmpIdx = tmpL<0; % outward normal vector
normVecbd(tmpIdx,:) = -normVecbd(tmpIdx,:);

% nu*<{grad u}n,[v^D]>
AcdBB = sum(DphiB.*normVecbd,2).*JumpB.*areabdFace; % nu*<{grad u^C}n, [v^D]>
Add0B = dot(normVecbd,JumpB,2).*areabdFace; % nu*<{grad u^D}n, [v^D]>

AcdBB = permute(AcdBB,[1 3 2]);
AcdBB = [AcdBB(:,:,1),AcdBB(:,:,2),AcdBB(:,:,3)];

Iu =  [elem(TB,:),NO+elem(TB,:),2*NO+elem(TB,:),3*NO+TB];
ii = [3*NO+repmat(TB,1,13),Iu];
jj = [Iu,3*NO+repmat(TB,1,13)];
ss = -nu_val*[AcdBB,Add0B,AcdBB,Add0B];
A = A + sparse(ii,jj,ss,DoF,DoF);

% rho/h<[u^D],[v^D]>
ii = 3*NO+TB;
jj = 3*NO+TB;
ss = nu_val*rho*dot(JumpB,JumpB,2).*sqrt(areabdFace);
A = A + sparse(ii,jj,ss,DoF,DoF);
B2 = B2 + sparse(TB,TB,ss,NT,NT);

% <{p},[v^D]>
ii = [3*NO+TB,3*NO+NT+TB];
jj = [3*NO+NT+TB,3*NO+TB];
ss = [Add0B,Add0B];
A = A + sparse(ii,jj,ss,DoF,DoF);

%% Assemble Right Hand Side

F = zeros(DoF,1);

[lambda,weight] = quadpts3(5);
phi = lambda; % linear bases
nQuad = size(lambda,1);
ft1 = zeros(NT,4); ft2 = zeros(NT,4); ft3 = zeros(NT,4);
f0 = zeros(NT,1); pt = zeros(NT,1);

NiF = size(inFace,1);
tmpL = double(T.face2elem(inFace,3));
face2localnodeL = [mod(tmpL+1,4), mod(tmpL+2,4), mod(tmpL+3,4)];
face2localnodeL = face2localnodeL + 4*(face2localnodeL == 0);
i1L = face2localnodeL(:,1);
i2L = face2localnodeL(:,2);
i3L = face2localnodeL(:,3);

tmpR = double(T.face2elem(inFace,4));
face2localnodeR = [mod(tmpR+1,4), mod(tmpR+2,4), mod(tmpR+3,4)];
face2localnodeR = face2localnodeR + 4*(face2localnodeR == 0);
i1R = face2localnodeR(:,1);
i2R = face2localnodeR(:,2);
i3R = face2localnodeR(:,3);


DphiL = [DphiL(:,:,1); DphiL(:,:,2); DphiL(:,:,3); DphiL(:,:,4)];
DphiL1 = DphiL((1:NiF)'+(i1L-1)*NiF,:);
DphiL2 = DphiL((1:NiF)'+(i2L-1)*NiF,:);
DphiL3 = DphiL((1:NiF)'+(i3L-1)*NiF,:);
DphiR = [DphiR(:,:,1); DphiR(:,:,2); DphiR(:,:,3); DphiR(:,:,4)];
DphiR1 = DphiR((1:NiF)'+(i1R-1)*NiF,:);
DphiR2 = DphiR((1:NiF)'+(i2R-1)*NiF,:);
DphiR3 = DphiR((1:NiF)'+(i3R-1)*NiF,:);

PhiDoFL = 2*(cross(DphiL2,DphiL3,2) ...
         + cross(DphiL3,DphiL1,2)...
         + cross(DphiL1,DphiL2,2));
PhiDoFL = dot(PhiDoFL,normVecin,2).*(areainFace/3);
PhiDoFR = 2*(cross(DphiR2,DphiR3,2) ...
         + cross(DphiR3,DphiR1,2)...
         + cross(DphiR1,DphiR2,2));
PhiDoFR = dot(PhiDoFR,normVecin,2).*(areainFace/3);

coefL = dot(JumpL,normVecin,2)/2.*areainFace;
coefR = dot(JumpR,normVecin,2)/2.*areainFace;

if TestType == 1
    for p = 1:nQuad
        % quadrature points in the x-y-z coordinate
        pxyz = lambda(p,1)*node(elem(:,1),:) ...
            + lambda(p,2)*node(elem(:,2),:) ...
            + lambda(p,3)*node(elem(:,3),:) ...
            + lambda(p,4)*node(elem(:,4),:);
        fp = pde.rhs(pxyz);
        pp = pde.exact_p(pxyz);
        for j = 1:4
            ft1(:,j) = ft1(:,j) + weight(p)*phi(p,j)*fp(:,1);
            ft2(:,j) = ft2(:,j) + weight(p)*phi(p,j)*fp(:,2);
            ft3(:,j) = ft3(:,j) + weight(p)*phi(p,j)*fp(:,3);
        end
        f0 = f0 + weight(p)*dot(pxyz-xT,fp,2).*volume;
        pt = pt + weight(p)*pp;
    end
    ft1 = ft1.*repmat(volume,1,4);
    ft2 = ft2.*repmat(volume,1,4);
    ft3 = ft3.*repmat(volume,1,4);
    
    f1 = accumarray(elem(:),ft1(:),[NO 1]);
    f2 = accumarray(elem(:),ft2(:),[NO 1]);
    f3 = accumarray(elem(:),ft3(:),[NO 1]);
    
    F(1:3*NO+NT) = [f1;f2;f3;f0];
end

if TestType == 2
    ftL = zeros(NiF,1); ftR = zeros(NiF,1);
    VALPHI1 = zeros(NiF,2);
    for p = 1:nQuad
        % quadrature points in the x-y-z coordinate
        pxyz = lambda(p,1)*node(elem(:,1),:) ...
            + lambda(p,2)*node(elem(:,2),:) ...
            + lambda(p,3)*node(elem(:,3),:) ...
            + lambda(p,4)*node(elem(:,4),:);
        pxyzL = pxyz(TL,:); pxyzR = pxyz(TR,:);
        fp = pde.rhs(pxyz); fpL = pde.rhs(pxyzL); fpR = pde.rhs(pxyzR);
        pp = pde.exact_p(pxyz);
        PhiL = 2*(phi(p,i1L)'.*cross(DphiL2,DphiL3,2) + ...
                 phi(p,i2L)'.*cross(DphiL3,DphiL1,2) + ...
                 phi(p,i3L)'.*cross(DphiL1,DphiL2,2));
        PhiR = 2*(phi(p,i1R)'.*cross(DphiR2,DphiR3,2) + ...
                 phi(p,i2R)'.*cross(DphiR3,DphiR1,2) + ...
                 phi(p,i3R)'.*cross(DphiR1,DphiR2,2));
        ftL = ftL + weight(p)*dot(fpL,PhiL,2).*volume(TL,:);
        ftR = ftR + weight(p)*dot(fpR,PhiR,2).*volume(TR,:);
        for j = 1:4
            ft1(:,j) = ft1(:,j) + weight(p)*phi(p,j)*fp(:,1);
            ft2(:,j) = ft2(:,j) + weight(p)*phi(p,j)*fp(:,2);
            ft3(:,j) = ft3(:,j) + weight(p)*phi(p,j)*fp(:,3);
        end
        pt = pt + weight(p)*pp;
    end
    
    VALPHI1(:,1) = ftL;
    VALPHI1(:,2) = ftR;

    fL = coefL.*ftL.*PhiDoFL + coefL.*ftR.*PhiDoFR; 
    fR = coefR.*ftR.*PhiDoFR + coefR.*ftL.*PhiDoFL;

    fL = accumarray(TL,fL,[NT 1]);
    fR = accumarray(TR,fR,[NT 1]);

    ft1 = ft1.*repmat(volume,1,4);
    ft2 = ft2.*repmat(volume,1,4);
    ft3 = ft3.*repmat(volume,1,4);

    f1 = accumarray(elem(:),ft1(:),[NO 1]);
    f2 = accumarray(elem(:),ft2(:),[NO 1]);
    f3 = accumarray(elem(:),ft3(:),[NO 1]);

    F(1:3*NO+NT) = [f1;f2;f3;(fL+fR)];
end

%% Solve the linear system

x = zeros(DoF,1);

bdNode = unique([T.bdFace(:,1);T.bdFace(:,2);T.bdFace(:,3)]);
isBdDoF = false(DoF,1);
isBdDoF([bdNode;NO+bdNode;2*NO+bdNode;end]) = true;
freeDoF = find(~isBdDoF);

u = pde.exact_u(node);
x(bdNode) = u(bdNode,1);
x(NO+bdNode) = u(bdNode,2);
x(2*NO+bdNode) = u(bdNode,3);
x(end) = pt(end);

F = F - A(:,isBdDoF)*x(isBdDoF);

x(freeDoF) = A(freeDoF,freeDoF)\F(freeDoF);

%% Check Numerical Solutions and Errors

diff_uC = x(1:3*NO)-u(:);
diff_uD = x(3*NO+1:3*NO+NT);

ph = x(3*NO+NT+1:3*NO+2*NT,1);

uC = zeros(NO,3);
uC(:,1) = x(1:NO); uC(:,2) = x(NO+1:2*NO); uC(:,3) = x(2*NO+1:3*NO);
uD = x(3*NO+1:3*NO+NT);

Du1 = Dphi(:,:,1).*uC(elem(:,1),1) + Dphi(:,:,2).*uC(elem(:,2),1)...
        + Dphi(:,:,3).*uC(elem(:,3),1) + Dphi(:,:,4).*uC(elem(:,4),1)...
        + uD*[1 0 0];
Du2 = Dphi(:,:,1).*uC(elem(:,1),2) + Dphi(:,:,2).*uC(elem(:,2),2)...
        + Dphi(:,:,3).*uC(elem(:,3),2) + Dphi(:,:,4).*uC(elem(:,4),2)...
        + uD*[0 1 0];
Du3 = Dphi(:,:,1).*uC(elem(:,1),3) + Dphi(:,:,2).*uC(elem(:,2),3)...
        + Dphi(:,:,3).*uC(elem(:,3),3) + Dphi(:,:,4).*uC(elem(:,4),3)...
        + uD*[0 0 1];
err_uCt = zeros(NT,1); err_pt = zeros(NT,1);

for p = 1:nQuad
    pxyz = lambda(p,1)*node(elem(:,1),:) ...
            + lambda(p,2)*node(elem(:,2),:) ...
            + lambda(p,3)*node(elem(:,3),:) ...
            + lambda(p,4)*node(elem(:,4),:);
    Du = pde.exact_gu(pxyz); pval = pde.exact_p(pxyz);
    err_uCt = err_uCt + weight(p)*(sum((Du1-Du(:,:,1)).^2,2)...
                            + sum((Du2-Du(:,:,2)).^2,2)...
                            + sum((Du3-Du(:,:,3)).^2,2)).*volume;
    err_pt = err_pt + weight(p)*(pval-ph).^2;
end


err_u = sqrt(sum(err_uCt) + (diff_uD'*B2*diff_uD)/nu_val);
err_p = sqrt(sum(err_pt.*volume));
err_axp = sqrt(sum((ph-pt).^2.*volume)); %added

fprintf('\n    ||u-u_h||_E : %.3e    ||P_0p-p_h||_0 : %.3e    ||p-p_h||_0 : %.3e\n\n',err_u,err_axp,err_p)
