%% MAIN_EGWG2: Modified Enriched Galerkin Method for Stokes equations
%
% MAIN_EGWG2 produces numerical solution using a modified pressure-robust
% Enriched Galerkin (EG) method for the Stokes equation in 2-dimensional
% domain:
%
%   - div(nu*grad(u)) + grad(p) = f in Omega
%                      - div(u) = 0 in Omega
%
% with a Dirichlet boundary condition. In the EG method, continuous
% piecewise linear functions and discontinuous piecewise constant functions
% are applied for the velocity and the pressure, respectively. Moreover,
% additional degrees of freedom are added to the velocity, and they are
% defined as discontinuous linear basis functions, that is, x - x_T for
% each element T. Using the same finite elements, the formulation is
% modified by using weak gradient and divergence operators.
%
% For pressure robust scheme, a divergence-preserving velocity
% reconstruction operator is applied to modify the body force assembling on
% the right hand side, so the same stiffness matrix will be used. The
% variable 'TestType' indicates the standard EG method (TestType = 1) or
% the pressure-robust EG method (TestType = 2).
%
% Dependencies (from iFEM by L. Chen):
%   auxstructure.m, squaremesh.m, gradbasis.m
%
% See also: Stokes2.m, as examples.
%
% References:
%   S. Lee and L. Mu, "A low-cost, penalty parameter-free, and
%     pressure-robust enriched Galerkin method for the Stokes equations,"
%     Computers & Mathematics with Applications, 2024.
%
% Authors: Seulip Lee and Lin Mu
%
clear
close all

%% Preliminaries

% Parameters
Deg = 1;
Num = 32;
pde = Stokes2;
rho = 1;
TestType = 1;
nu_val = pde.nu;

% Mesh generation
[node,elem] = squaremesh([0,1,0,1],1/Num);
T = auxstructure(elem);
[Dphi,area] = gradbasis(node,elem);


% Number of elements, edges, nodes
NT = size(elem,1);
NE = size(T.edge,1);
NO = size(node,1);

% Degrees of freedom
DoF_u = (2*NO+NT);
DoF_p = NT;
DoF = DoF_u + DoF_p;

%% Weak Derivatives

TID = (1:NT)'; idxNe = T.neighbor ~= TID;
xmid = (node(T.edge(:,1),:)+node(T.edge(:,2),:))/2;
xT = (node(elem(:,1),:)+node(elem(:,2),:)+node(elem(:,3),:))/3;
lengEdge = sqrt(sum((node(T.edge(:,1),:)-node(T.edge(:,2),:)).^2,2));
normVec = [(node(T.edge(:,2),2)-node(T.edge(:,1),2)),...
           (node(T.edge(:,1),1)-node(T.edge(:,2),1))];
normVec = normVec./lengEdge;

el2ed1 = T.elem2edge(:,1); el2ed2 = T.elem2edge(:,2); el2ed3 = T.elem2edge(:,3);
xmidT1 = xmid(el2ed1,:); xmidT2 = xmid(el2ed2,:); xmidT3 = xmid(el2ed3,:);
outVec1 = xmidT1-xT; outVec2 = xmidT2-xT; outVec3 = xmidT3-xT;
flag1 = dot(outVec1,normVec(el2ed1,:),2);
flag2 = dot(outVec2,normVec(el2ed2,:),2);
flag3 = dot(outVec3,normVec(el2ed3,:),2);
flag1 = 2*(flag1>0)-1; flag2 = 2*(flag2>0)-1; flag3 = 2*(flag3>0)-1;

lengbdEdge = sqrt(sum((node(T.bdEdge(:,1),:)-node(T.bdEdge(:,2),:)).^2,2));
normbdVec = [(node(T.bdEdge(:,2),2)-node(T.bdEdge(:,1),2)),...
             (node(T.bdEdge(:,1),1)-node(T.bdEdge(:,2),1))];
normbdVec = normbdVec./lengbdEdge;
xmidbd = (node(T.bdEdge(:,1),:)+node(T.bdEdge(:,2),:))/2;
outVecbd = xmidbd-xT(T.bdEdge2elem,:);
flagbd = dot(outVecbd,normbdVec,2);
flagbd = 2*(flagbd>0)-1;

wDxT = (normVec(el2ed1,:).*(xmid(el2ed1,1)-xT(:,1)).*lengEdge(el2ed1,:).*flag1...
    + normVec(el2ed2,:).*(xmid(el2ed2,1)-xT(:,1)).*lengEdge(el2ed2,:).*flag2...
    + normVec(el2ed3,:).*(xmid(el2ed3,1)-xT(:,1)).*lengEdge(el2ed3,:).*flag3)./(2*area);
wDyT = (normVec(el2ed1,:).*(xmid(el2ed1,2)-xT(:,2)).*lengEdge(el2ed1,:).*flag1...
    + normVec(el2ed2,:).*(xmid(el2ed2,2)-xT(:,2)).*lengEdge(el2ed2,:).*flag2...
    + normVec(el2ed3,:).*(xmid(el2ed3,2)-xT(:,2)).*lengEdge(el2ed3,:).*flag3)./(2*area);
wDxT(T.bdEdge2elem,:) = wDxT(T.bdEdge2elem,:) - normbdVec.*(xmidbd(:,1)-xT(T.bdEdge2elem,1))...
                                .*lengbdEdge.*flagbd./(2*area(T.bdEdge2elem,:));
wDyT(T.bdEdge2elem,:) = wDyT(T.bdEdge2elem,:) - normbdVec.*(xmidbd(:,2)-xT(T.bdEdge2elem,2))...
                                .*lengbdEdge.*flagbd./(2*area(T.bdEdge2elem,:));

wDxN(:,:,1) = normVec(el2ed1,:).*(xmid(el2ed1,1)-xT(T.neighbor(:,1),1))...
                .*lengEdge(el2ed1,:).*idxNe(:,1).*flag1./(2*area);
wDxN(:,:,2) = normVec(el2ed2,:).*(xmid(el2ed2,1)-xT(T.neighbor(:,2),1))...
                .*lengEdge(el2ed2,:).*idxNe(:,2).*flag2./(2*area);
wDxN(:,:,3) = normVec(el2ed3,:).*(xmid(el2ed3,1)-xT(T.neighbor(:,3),1))...
                .*lengEdge(el2ed3,:).*idxNe(:,3).*flag3./(2*area);

wDyN(:,:,1) = normVec(el2ed1,:).*(xmid(el2ed1,2)-xT(T.neighbor(:,1),2))...
                .*lengEdge(el2ed1,:).*idxNe(:,1).*flag1./(2*area);
wDyN(:,:,2) = normVec(el2ed2,:).*(xmid(el2ed2,2)-xT(T.neighbor(:,2),2))...
                .*lengEdge(el2ed2,:).*idxNe(:,2).*flag2./(2*area);
wDyN(:,:,3) = normVec(el2ed3,:).*(xmid(el2ed3,2)-xT(T.neighbor(:,3),2))...
                .*lengEdge(el2ed3,:).*idxNe(:,3).*flag3./(2*area);

wDivN = wDxN(:,1,:) + wDyN(:,2,:);
wDivT = wDxT(:,1) + wDyT(:,2);

%% Assemble Element Terms

A = sparse(DoF,DoF);
B1 = sparse(2*NO,2*NO); TN = T.neighbor;
for i = 1:3
    for j = 1:3
        % nu*(grad u^C, grad v^C)
        Acc00 = nu_val*dot(Dphi(:,:,i),Dphi(:,:,j),2).*area; % 0-0
        AddNN = nu_val*(dot(wDxN(:,:,i),wDxN(:,:,j),2)+...
                        dot(wDyN(:,:,i),wDyN(:,:,j),2)).*area; % N-N
        Acd0N = nu_val*[dot(wDxN(:,:,i),Dphi(:,:,j),2),...
                        dot(wDyN(:,:,i),Dphi(:,:,j),2)].*area; % 0-N
        AcdN0 = nu_val*[dot(wDxN(:,:,j),Dphi(:,:,i),2),...
                        dot(wDyN(:,:,j),Dphi(:,:,i),2)].*area; % N-0

        ii = [elem(:,i),NO+elem(:,i),2*NO+TN(:,i),2*NO+TN(:,i),2*NO+TN(:,i),elem(:,i),NO+elem(:,i)];
        jj = [elem(:,j),NO+elem(:,j),2*NO+TN(:,j),elem(:,j),NO+elem(:,j),2*NO+TN(:,j),2*NO+TN(:,j)];
        ss = [Acc00,Acc00,AddNN,Acd0N,AcdN0];
        A = A + sparse(ii,jj,ss,DoF,DoF);
        B1 = B1 + sparse([elem(:,i);NO+elem(:,i)],[elem(:,j);NO+elem(:,j)],[Acc00;Acc00],2*NO,2*NO);
    end

    Acd0T = nu_val*[dot(wDxT,Dphi(:,:,i),2),dot(wDyT,Dphi(:,:,i),2)].*area;
    AddNT = nu_val*(dot(wDxT,wDxN(:,:,i),2)+dot(wDyT,wDyN(:,:,i),2)).*area;
    

    ii = [elem(:,i),NO+elem(:,i),2*NO+TID,2*NO+TID,2*NO+TN(:,i),2*NO+TID];
    jj = [2*NO+TID,2*NO+TID,elem(:,i),NO+elem(:,i),2*NO+TID,2*NO+TN(:,i)];
    ss = [Acd0T,Acd0T,AddNT,AddNT];
    A = A + sparse(ii,jj,ss,DoF,DoF);

    Acp = Dphi(:,:,i).*area; % -(div u^C, q) and -(div v^C, p)
    ii = [elem(:,i),NO+elem(:,i),2*NO+NT+TID,2*NO+NT+TID];
    jj = [2*NO+NT+TID,2*NO+NT+TID,elem(:,i),NO+elem(:,i)];
    ss = [-Acp,-Acp];
    
    A = A + sparse(ii,jj,ss,DoF,DoF);
end

AddTT = nu_val*(dot(wDxT,wDxT,2)+dot(wDyT,wDyT,2)).*area;
Adp = 2*area;
A = A + sparse([2*NO+TID,2*NO+TID,2*NO+NT+TID], ...
    [2*NO+TID,2*NO+NT+TID,2*NO+TID],[AddTT,-Adp,-Adp],DoF,DoF);

%% Assemble Interior Edge Terms

B2 = sparse(NT,NT);

bdEdge = find(T.edge2elem(:,1)==T.edge2elem(:,2));
inEdge = setdiff((1:NE)',bdEdge);

xE = (node(T.edge(inEdge,1),:)+node(T.edge(inEdge,2),:))/2;

lenginEdge = sqrt(sum((node(T.edge(inEdge,1),:)-node(T.edge(inEdge,2),:)).^2,2));
normVecin = [(node(T.edge(inEdge,2),2)-node(T.edge(inEdge,1),2)),...
             (node(T.edge(inEdge,1),1)-node(T.edge(inEdge,2),1))];
normVecin = normVecin./lenginEdge;

TL = double(T.edge2elem(inEdge,1));
TR = double(T.edge2elem(inEdge,2));

tmpL = dot(normVecin,xT(TR,:)-xT(TL,:),2);
tmpIdx = tmpL<0;
normVecin(tmpIdx,:) = -normVecin(tmpIdx,:);

DphiL = Dphi(TL,:,:); DphiR = Dphi(TR,:,:);
JumpL = xE - xT(TL,:); JumpR = xE - xT(TR,:);

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

% nu*rho/h*<[u^D],[v^D]>
ii = [2*NO+TL,2*NO+TL,2*NO+TR,2*NO+TR];
jj = [2*NO+TL,2*NO+TR,2*NO+TL,2*NO+TR];
ss = nu_val*rho*[dot(JumpL,JumpL,2),-dot(JumpL,JumpR,2),...
    -dot(JumpR,JumpL,2),dot(JumpR,JumpR,2)];
A = A + sparse(ii,jj,ss,DoF,DoF);
B2 = B2 + sparse([TL,TL,TR,TR],[TL,TR,TL,TR],ss,NT,NT);

% <{p},[v^D]> & <{q},[u^D]>
ii = [2*NO+TL,2*NO+TL,2*NO+TR,2*NO+TR,...
    2*NO+NT+TL,2*NO+NT+TR,2*NO+NT+TL,2*NO+NT+TR];
jj = [2*NO+NT+TL,2*NO+NT+TR,2*NO+NT+TL,2*NO+NT+TR...
    2*NO+TL,2*NO+TL,2*NO+TR,2*NO+TR];
ss = [Add0L,Add0L,Add0R,Add0R,Add0L,Add0L,Add0R,Add0R]/2;
A = A + sparse(ii,jj,ss,DoF,DoF);

%% Assemble Boundary Edge Terms

T.bdEdge = double(T.bdEdge);

xE = (node(T.bdEdge(:,1),:)+node(T.bdEdge(:,2),:))/2;
lengbdEdge = sqrt(sum((node(T.bdEdge(:,1),:)-node(T.bdEdge(:,2),:)).^2,2));
normVecbd = [(node(T.bdEdge(:,2),2)-node(T.bdEdge(:,1),2)),...
             (node(T.bdEdge(:,1),1)-node(T.bdEdge(:,2),1))];
normVecbd = normVecbd./lengbdEdge;

TB = T.bdEdge2elem;
DphiB = Dphi(TB,:,:);
JumpB = xE - xT(TB,:);

tmpL = dot(normVecbd,JumpB,2);
tmpIdx = tmpL<0;
normVecbd(tmpIdx,:) = -normVecbd(tmpIdx,:);

% nu*<{grad u}n,[v^D]>
AcdBB = sum(DphiB.*normVecbd,2).*JumpB.*lengbdEdge;
Add0B = dot(normVecbd,JumpB,2).*lengbdEdge;

AcdBB = permute(AcdBB,[1 3 2]);
AcdBB = [AcdBB(:,:,1),AcdBB(:,:,2)];

% rho/h<[u^D],[v^D]>
ii = 2*NO+TB;
jj = 2*NO+TB;
ss = nu_val*rho*dot(JumpB,JumpB,2);
A = A + sparse(ii,jj,ss,DoF,DoF);
B2 = B2 + sparse(TB,TB,ss,NT,NT);

% <{p},[v^D]>
ii = [2*NO+TB,2*NO+NT+TB];
jj = [2*NO+NT+TB,2*NO+TB];
ss = [Add0B,Add0B];
A = A + sparse(ii,jj,ss,DoF,DoF);

%% Assemble Right Hand Side

F = zeros(DoF,1);
[lambda,weight] = quadpts(9);
phi = lambda;
nQuad = size(lambda,1);
ft1 = zeros(NT,3); ft2 = zeros(NT,3);
f0 = zeros(NT,1); pt = zeros(NT,1);

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

end

if TestType == 2

    NiE = size(inEdge,1);
    ftL = zeros(NiE,1); ftR = zeros(NiE,1);
    
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

u = pde.exact_u(node);
x(bdNode) = u(bdNode,1);
x(NO+bdNode) = u(bdNode,2);
x(end) = pt(end);

F = F - A(:,isBdDoF)*x(isBdDoF);

x(freeDoF) = A(freeDoF,freeDoF)\F(freeDoF);

%% Compute Errors

diff_uC = x(1:2*NO)-u(:);

ph = x(2*NO+NT+1:2*NO+2*NT);

uC = zeros(NO,2); uC(:,1) = x(1:NO); uC(:,2) = x(NO+1:2*NO);
uD = x(2*NO+1:2*NO+NT);
Du1 = Dphi(:,:,1).*uC(elem(:,1),1) + Dphi(:,:,2).*uC(elem(:,2),1)...
        + Dphi(:,:,3).*uC(elem(:,3),1) + uD*[1 0];
Du2 = Dphi(:,:,1).*uC(elem(:,1),2) + Dphi(:,:,2).*uC(elem(:,2),2)...
        + Dphi(:,:,3).*uC(elem(:,3),2) + uD*[0 1];

err_uCt = zeros(NT,1); err_pt = zeros(NT,1);
for p = 1:nQuad
    pxy = lambda(p,1)*node(elem(:,1),:) ...
            + lambda(p,2)*node(elem(:,2),:) ...
            + lambda(p,3)*node(elem(:,3),:);
    Du = pde.exact_gu(pxy); pval = pde.exact_p(pxy);
    err_uCt = err_uCt + weight(p)*(sum((Du1-Du(:,:,1)).^2,2)...
                            + sum((Du2-Du(:,:,2)).^2,2)).*area;
    err_pt = err_pt + weight(p)*(pval-ph).^2;
end

err_u = sqrt(sum(err_uCt)+(uD'*B2*uD)/nu_val);
err_p = sqrt(sum(err_pt.*area));
err_axp = sqrt(sum((ph-pt).^2.*area));

fprintf('\n    ||u-u_h||_E : %.3e    ||P_0p-p_h||_0 : %.3e    ||p-p_h||_0 : %.3e\n\n',err_u,err_axp,err_p)

%% Plot Numerical Solutions

uhT = zeros(NO,1); vhT = zeros(NO,1); phT = zeros(NO,1);
count = zeros(NO,1);
for T = 1:NT
    uDp(:,T) = x(2*NO+T)*(node(elem(T,:),1)-xT(T,1));
    vDp(:,T) = x(2*NO+T)*(node(elem(T,:),2)-xT(T,2));
end
uDp = uDp(:);
vDp = vDp(:);

elem_tr = elem';
P = node(elem_tr(:),:);
uC = x(elem_tr(:),:);
vC = x(elem_tr(:)+NO,:);
tri=reshape(1:size(P,1),[3 size(P,1)/3]);
colormap jet
trisurf(tri',P(:,1),P(:,2),uC+uDp,'EdgeColor', 'none');
view(2); axis square;
ax = gca; ax.FontSize = 20;
box on
c = colorbar; c.FontSize = 20; c.Location = 'southoutside';
figure
colormap jet
trisurf(tri',P(:,1),P(:,2),vC+vDp,'EdgeColor', 'none');
view(2); axis square;
ax = gca; ax.FontSize = 20;
box on
c = colorbar; c.FontSize = 20; c.Location = 'southoutside';
uCD = u(elem_tr(:),1);
vCD = u(elem_tr(:),2);

% Plot pressure
figure
ph = x(2*NO+NT+1:2*NO+2*NT,1);
colormap jet
trisurf(tri',P(:,1),P(:,2),[ph';ph';ph'],'EdgeColor', 'none');
view(2); axis square;
ax = gca; ax.FontSize = 20;
box on
c = colorbar; c.FontSize = 20; c.Location = 'southoutside';

figure
UDD = x(2*NO+1:2*NO+NT,1);
trisurf(tri',P(:,1),P(:,2),[UDD';UDD';UDD'],'EdgeColor', 'none');
