%% MAIN_EGTimeBE3: Enriched Galerkin Method for Time-Dependent Stokes (3D, BE)
%
% Solves the time-dependent Stokes equations in a 3-dimensional domain:
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
%   cubemesh.m, auxstructure3.m, gradbasis3.m, quadpts3.m, mycross.m, myunique.m
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

Num = 16; % 1/Num = h, dt = 1/Num
rho1 = 3; dt = 1/Num; T_end = 1; time_it = T_end/dt;
TestType = 3; % 1: standard / 2: PR body force only / 3: PR body force + mass term
assert(ismember(TestType,[1 2 3]), 'TestType must be 1, 2, or 3.');
pde = Stokes_Time3; mu_fun = pde.mu;

[node,elem] = cubemesh([0,1,0,1,0,1],1/Num);
T = auxstructure3(elem);
[Dphi,volume] = gradbasis3(node,elem);
xT = (node(elem(:,1),:)+node(elem(:,2),:)+node(elem(:,3),:)+node(elem(:,4),:))/4;
xF = (node(T.face(:,1),:)+node(T.face(:,2),:)+node(T.face(:,3),:))/3;

NT = size(elem,1); NF = size(T.face,1); NO = size(node,1);
DoF_u = (3*NO+NT); DoF_p = NT; DoF = DoF_u + DoF_p;
TID = (1:NT)';

mu_elem = mu_fun(xT);

[lambda,weight] = quadpts3(5); phi = lambda; nQuad = size(lambda,1);

%% Assemble Element Terms

A = sparse(DoF,DoF);
localC = [2 1 1 1; 1 2 1 1; 1 1 2 1; 1 1 1 2]/20;

for i = 1:4
    for j = 1:4
        Acc = mu_elem.*dot(Dphi(:,:,i),Dphi(:,:,j),2).*volume;
        A = A + sparse([elem(:,i);NO+elem(:,i);2*NO+elem(:,i)],...
                       [elem(:,j);NO+elem(:,j);2*NO+elem(:,j)],...
                       [Acc;Acc;Acc],DoF,DoF);
    end
    Acp = Dphi(:,:,i).*volume;
    Acd = mu_elem.*Acp;
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

Adp = 3*volume;
Add = mu_elem.*Adp;
A = A + sparse([3*NO+TID,3*NO+TID,3*NO+NT+TID],...
               [3*NO+TID,3*NO+NT+TID,3*NO+TID],[Add,-Adp,-Adp],DoF,DoF);

%% Assemble Interior Face Terms

B = sparse(NT,NT);

bdFace = find(T.face2elem(:,1)==T.face2elem(:,2));
inFace = setdiff((1:NF)',bdFace);
xFin = xF(inFace,:);

r12 = node(T.face(inFace,2),:) - node(T.face(inFace,1),:);
r13 = node(T.face(inFace,3),:) - node(T.face(inFace,1),:);
normVecin = cross(r12,r13,2);
areainFace = vecnorm(normVecin,2,2)/2;
normVecin = normVecin./(areainFace*2);

TL = double(T.face2elem(inFace,1));
TR = double(T.face2elem(inFace,2));
mu_inFace = (mu_elem(TL)+mu_elem(TR))/2;

tmpL = dot(normVecin,xT(TR,:)-xT(TL,:),2);
tmpIdx = tmpL<0;
normVecin(tmpIdx,:) = -normVecin(tmpIdx,:);

DphiL = Dphi(TL,:,:); DphiR = Dphi(TR,:,:);
JumpL = xFin - xT(TL,:); JumpR = xFin - xT(TR,:);

AcdLL = sum(DphiL.*normVecin,2).*JumpL.*areainFace;
AcdRL = sum(DphiR.*normVecin,2).*JumpL.*areainFace;
Add0L = dot(normVecin,JumpL,2).*areainFace;
AcdRR = -sum(DphiR.*normVecin,2).*JumpR.*areainFace;
AcdLR = -sum(DphiL.*normVecin,2).*JumpR.*areainFace;
Add0R = -dot(normVecin,JumpR,2).*areainFace;

AcdLL = permute(AcdLL,[1 3 2]); AcdRL = permute(AcdRL,[1 3 2]);
AcdRR = permute(AcdRR,[1 3 2]); AcdLR = permute(AcdLR,[1 3 2]);
AcdLL = [AcdLL(:,:,1),AcdLL(:,:,2),AcdLL(:,:,3)];
AcdRL = [AcdRL(:,:,1),AcdRL(:,:,2),AcdRL(:,:,3)];
AcdLR = [AcdLR(:,:,1),AcdLR(:,:,2),AcdLR(:,:,3)];
AcdRR = [AcdRR(:,:,1),AcdRR(:,:,2),AcdRR(:,:,3)];

valL = [AcdLL,AcdRL,Add0L,Add0L]/2;
valR = [AcdLR,AcdRR,Add0R,Add0R]/2;

Iu = [elem(TL,:),NO+elem(TL,:),2*NO+elem(TL,:),...
      elem(TR,:),NO+elem(TR,:),2*NO+elem(TR,:),...
      3*NO+TL,3*NO+TR];
ii = [3*NO+repmat(TL,1,26),Iu,3*NO+repmat(TR,1,26),Iu];
jj = [Iu,3*NO+repmat(TL,1,26),Iu,3*NO+repmat(TR,1,26)];
ss = -mu_inFace.*[valL,valL,valR,valR];
A = A + sparse(ii,jj,ss,DoF,DoF);

% mu*rho1*sqrt(|f|)*<[u^D],[v^D]>
ii = [3*NO+TL,3*NO+TL,3*NO+TR,3*NO+TR];
jj = [3*NO+TL,3*NO+TR,3*NO+TL,3*NO+TR];
ss = mu_inFace*rho1.*[dot(JumpL,JumpL,2),-dot(JumpL,JumpR,2),...
     -dot(JumpR,JumpL,2),dot(JumpR,JumpR,2)].*sqrt(areainFace);
A = A + sparse(ii,jj,ss,DoF,DoF);
B = B + sparse([TL,TL,TR,TR],[TL,TR,TL,TR],ss,NT,NT);

% <{p},[v^D]> & <{q},[u^D]>
ii = [3*NO+TL,3*NO+TL,3*NO+TR,3*NO+TR,...
      3*NO+NT+TL,3*NO+NT+TR,3*NO+NT+TL,3*NO+NT+TR];
jj = [3*NO+NT+TL,3*NO+NT+TR,3*NO+NT+TL,3*NO+NT+TR,...
      3*NO+TL,3*NO+TL,3*NO+TR,3*NO+TR];
ss = [Add0L,Add0L,Add0R,Add0R,Add0L,Add0L,Add0R,Add0R]/2;
A = A + sparse(ii,jj,ss,DoF,DoF);

%% Assemble Boundary Face Terms

T.bdFace = double(T.bdFace);
xFbd = (node(T.bdFace(:,1),:)+node(T.bdFace(:,2),:)+node(T.bdFace(:,3),:))/3;
r12 = node(T.bdFace(:,2),:) - node(T.bdFace(:,1),:);
r13 = node(T.bdFace(:,3),:) - node(T.bdFace(:,1),:);
normVecbd = cross(r13,r12,2);
areabdFace = vecnorm(normVecbd,2,2)/2;
normVecbd = normVecbd./(areabdFace*2);

TB = T.bdFace2elem;
mu_bd = mu_elem(TB);
DphiB = Dphi(TB,:,:);
JumpB = xFbd - xT(TB,:);
tmpL = dot(normVecbd,JumpB,2);
tmpIdx = tmpL<0;
normVecbd(tmpIdx,:) = -normVecbd(tmpIdx,:);

AcdBB = sum(DphiB.*normVecbd,2).*JumpB.*areabdFace;
Add0B = dot(normVecbd,JumpB,2).*areabdFace;
AcdBB = permute(AcdBB,[1 3 2]);
AcdBB = [AcdBB(:,:,1),AcdBB(:,:,2),AcdBB(:,:,3)];
Iu = [elem(TB,:),NO+elem(TB,:),2*NO+elem(TB,:),3*NO+TB];
ii = [3*NO+repmat(TB,1,13),Iu];
jj = [Iu,3*NO+repmat(TB,1,13)];
ss = -mu_bd.*[AcdBB,Add0B,AcdBB,Add0B];
A = A + sparse(ii,jj,ss,DoF,DoF);

ii = 3*NO+TB; jj = 3*NO+TB;
ss = mu_bd*rho1.*dot(JumpB,JumpB,2).*sqrt(areabdFace);
A = A + sparse(ii,jj,ss,DoF,DoF);
B = B + sparse(TB,TB,ss,NT,NT);

ii = [3*NO+TB,3*NO+NT+TB]; jj = [3*NO+NT+TB,3*NO+TB];
ss = [Add0B,Add0B];
A = A + sparse(ii,jj,ss,DoF,DoF);

%% Basis Functions for RT0 Reconstruction (needed for TestType 2 & 3)

NiF = size(inFace,1);

tmpL = double(T.face2elem(inFace,3));
face2localnodeL = [mod(tmpL+1,4), mod(tmpL+2,4), mod(tmpL+3,4)];
face2localnodeL = face2localnodeL + 4*(face2localnodeL == 0);
i1L = face2localnodeL(:,1); i2L = face2localnodeL(:,2); i3L = face2localnodeL(:,3);

tmpR = double(T.face2elem(inFace,4));
face2localnodeR = [mod(tmpR+1,4), mod(tmpR+2,4), mod(tmpR+3,4)];
face2localnodeR = face2localnodeR + 4*(face2localnodeR == 0);
i1R = face2localnodeR(:,1); i2R = face2localnodeR(:,2); i3R = face2localnodeR(:,3);

DphiL = [DphiL(:,:,1); DphiL(:,:,2); DphiL(:,:,3); DphiL(:,:,4)];
DphiL1 = DphiL((1:NiF)'+(i1L-1)*NiF,:);
DphiL2 = DphiL((1:NiF)'+(i2L-1)*NiF,:);
DphiL3 = DphiL((1:NiF)'+(i3L-1)*NiF,:);
DphiR = [DphiR(:,:,1); DphiR(:,:,2); DphiR(:,:,3); DphiR(:,:,4)];
DphiR1 = DphiR((1:NiF)'+(i1R-1)*NiF,:);
DphiR2 = DphiR((1:NiF)'+(i2R-1)*NiF,:);
DphiR3 = DphiR((1:NiF)'+(i3R-1)*NiF,:);

PhiDoFL = 2*(cross(DphiL2,DphiL3,2)+cross(DphiL3,DphiL1,2)+cross(DphiL1,DphiL2,2));
PhiDoFL = dot(PhiDoFL,normVecin,2).*(areainFace/3);
PhiDoFR = 2*(cross(DphiR2,DphiR3,2)+cross(DphiR3,DphiR1,2)+cross(DphiR1,DphiR2,2));
PhiDoFR = dot(PhiDoFR,normVecin,2).*(areainFace/3);

coefL = dot(JumpL,normVecin,2)/2.*areainFace;
coefR = dot(JumpR,normVecin,2)/2.*areainFace;

%% Assemble Velocity Mass Matrix M (for the Backward Euler (u,v)/dt term)

M = sparse(DoF_u,DoF_u);

for i = 1:4
    for j = 1:4
        % (v^C,w^C): always assembled directly, same for every TestType
        Mcc = localC(i,j)*volume;
        M = M + sparse([elem(:,i);NO+elem(:,i);2*NO+elem(:,i)],...
                       [elem(:,j);NO+elem(:,j);2*NO+elem(:,j)],...
                       [Mcc;Mcc;Mcc],DoF_u,DoF_u);
    end

    if TestType ~= 3
        % (v^C,w^D) and (v^D,w^C), assembled directly (mass term unmodified)
        Mcd = zeros(NT,3);
        for p = 1:nQuad
            pxyz = lambda(p,1)*node(elem(:,1),:) + lambda(p,2)*node(elem(:,2),:) ...
                 + lambda(p,3)*node(elem(:,3),:) + lambda(p,4)*node(elem(:,4),:);
            Mcd = Mcd + weight(p)*(pxyz-xT)*lambda(p,i).*volume;
        end
        ii = [elem(:,i),NO+elem(:,i),2*NO+elem(:,i),3*NO+TID,3*NO+TID,3*NO+TID];
        jj = [3*NO+TID,3*NO+TID,3*NO+TID,elem(:,i),NO+elem(:,i),2*NO+elem(:,i)];
        ss = [Mcd,Mcd];
        M = M + sparse(ii,jj,ss,DoF_u,DoF_u);
    end
end

if TestType ~= 3
    % (v^D,w^D), assembled directly (mass term unmodified)
    Mdd = zeros(NT,1);
    for p = 1:nQuad
        pxyz = lambda(p,1)*node(elem(:,1),:) + lambda(p,2)*node(elem(:,2),:) ...
             + lambda(p,3)*node(elem(:,3),:) + lambda(p,4)*node(elem(:,4),:);
        Mdd = Mdd + weight(p)*dot(pxyz-xT,pxyz-xT,2).*volume;
    end
    M = M + sparse(3*NO+TID,3*NO+TID,Mdd,DoF_u,DoF_u);
else
    % (v^C,w^D) and (v^D,w^D), via the RT0 velocity reconstruction
    R3d = sparse(NF,NF);
    Dcdt1L = zeros(NiF,4); Dcdt2L = zeros(NiF,4); Dcdt3L = zeros(NiF,4);
    Dcdt1R = zeros(NiF,4); Dcdt2R = zeros(NiF,4); Dcdt3R = zeros(NiF,4);

    % Face k is opposite local node k; face nodes = setdiff(1:4,k)
    faceLocalNodes = {[2,3,4],[1,3,4],[1,2,4],[1,2,3]};

    for p = 1:nQuad
        PhiL = 2*(phi(p,i1L)'.*cross(DphiL2,DphiL3,2) + ...
                  phi(p,i2L)'.*cross(DphiL3,DphiL1,2) + ...
                  phi(p,i3L)'.*cross(DphiL1,DphiL2,2));
        PhiR = 2*(phi(p,i1R)'.*cross(DphiR2,DphiR3,2) + ...
                  phi(p,i2R)'.*cross(DphiR3,DphiR1,2) + ...
                  phi(p,i3R)'.*cross(DphiR1,DphiR2,2));
        for j = 1:4
            Dcdt1L(:,j) = Dcdt1L(:,j) + weight(p)*phi(p,j)*PhiL(:,1).*volume(TL);
            Dcdt2L(:,j) = Dcdt2L(:,j) + weight(p)*phi(p,j)*PhiL(:,2).*volume(TL);
            Dcdt3L(:,j) = Dcdt3L(:,j) + weight(p)*phi(p,j)*PhiL(:,3).*volume(TL);
            Dcdt1R(:,j) = Dcdt1R(:,j) + weight(p)*phi(p,j)*PhiR(:,1).*volume(TR);
            Dcdt2R(:,j) = Dcdt2R(:,j) + weight(p)*phi(p,j)*PhiR(:,2).*volume(TR);
            Dcdt3R(:,j) = Dcdt3R(:,j) + weight(p)*phi(p,j)*PhiR(:,3).*volume(TR);
        end

        Phi3d = zeros(NT,3,4);
        for k = 1:4
            fn = faceLocalNodes{k};
            Phi3d(:,:,k) = 2*(phi(p,fn(1))*cross(Dphi(:,:,fn(2)),Dphi(:,:,fn(3)),2) + ...
                              phi(p,fn(2))*cross(Dphi(:,:,fn(3)),Dphi(:,:,fn(1)),2) + ...
                              phi(p,fn(3))*cross(Dphi(:,:,fn(1)),Dphi(:,:,fn(2)),2));
        end
        for i = 1:4
            for j = 1:4
                L2prod = weight(p)*dot(Phi3d(:,:,i),Phi3d(:,:,j),2).*volume;
                R3d = R3d + sparse(double(T.elem2face(:,i)),double(T.elem2face(:,j)),L2prod,NF,NF);
            end
        end
    end

    Dcd1L = [coefL.*PhiDoFL.*Dcdt1L, coefL.*PhiDoFR.*Dcdt1R];
    Dcd2L = [coefL.*PhiDoFL.*Dcdt2L, coefL.*PhiDoFR.*Dcdt2R];
    Dcd3L = [coefL.*PhiDoFL.*Dcdt3L, coefL.*PhiDoFR.*Dcdt3R];
    Dcd1R = [coefR.*PhiDoFL.*Dcdt1L, coefR.*PhiDoFR.*Dcdt1R];
    Dcd2R = [coefR.*PhiDoFL.*Dcdt2L, coefR.*PhiDoFR.*Dcdt2R];
    Dcd3R = [coefR.*PhiDoFL.*Dcdt3L, coefR.*PhiDoFR.*Dcdt3R];

    ii = [3*NO+repmat(TL,1,8),3*NO+repmat(TL,1,8),3*NO+repmat(TL,1,8),...
          3*NO+repmat(TR,1,8),3*NO+repmat(TR,1,8),3*NO+repmat(TR,1,8)];
    jj = [elem(TL,:),elem(TR,:),NO+elem(TL,:),NO+elem(TR,:),2*NO+elem(TL,:),2*NO+elem(TR,:),...
          elem(TL,:),elem(TR,:),NO+elem(TL,:),NO+elem(TR,:),2*NO+elem(TL,:),2*NO+elem(TR,:)];
    ss = [Dcd1L,Dcd2L,Dcd3L,Dcd1R,Dcd2R,Dcd3R];
    M = M + sparse(ii,jj,ss,DoF_u,DoF_u) + sparse(jj,ii,ss,DoF_u,DoF_u);

    ss_S = zeros(NT,4);
    for k = 1:4
        fn = faceLocalNodes{k};
        j1 = fn(1); j2 = fn(2); j3 = fn(3);
        xFk = (node(elem(:,j1),:)+node(elem(:,j2),:)+node(elem(:,j3),:))/3;
        r12k = node(elem(:,j2),:)-node(elem(:,j1),:);
        r13k = node(elem(:,j3),:)-node(elem(:,j1),:);
        nVeck = cross(r12k,r13k,2);
        aFk = vecnorm(nVeck,2,2)/2;
        nVeck = nVeck./(2*aFk);
        flipk = dot(nVeck, xFk - node(elem(:,k),:), 2) < 0;
        nVeck(flipk,:) = -nVeck(flipk,:);
        DpJ1 = Dphi(:,:,j1); DpJ2 = Dphi(:,:,j2); DpJ3 = Dphi(:,:,j3);
        PhiDoFTk = 2*(cross(DpJ1,DpJ2,2)+cross(DpJ2,DpJ3,2)+cross(DpJ3,DpJ1,2));
        PhiDoFTk = dot(PhiDoFTk, nVeck, 2).*(aFk/3);
        coefTk = dot(xFk-xT, nVeck, 2)/2.*aFk;
        ss_S(:,k) = coefTk.*PhiDoFTk;
    end
    S3d = sparse(repmat(TID,1,4), double(T.elem2face), ss_S, NT, NF);
    R3d(bdFace,:) = R3d(bdFace,:)*0; R3d(:,bdFace) = R3d(:,bdFace)*0;
    R3d = S3d*R3d*S3d';
    M(3*NO+1:3*NO+NT, 3*NO+1:3*NO+NT) = M(3*NO+1:3*NO+NT, 3*NO+1:3*NO+NT) + R3d;
end

A(1:DoF_u,1:DoF_u) = A(1:DoF_u,1:DoF_u) + M/dt;

%% Boundary Condition Setup

bdNode = unique([T.bdFace(:,1);T.bdFace(:,2);T.bdFace(:,3)]);
isBdDoF = false(DoF,1);
isBdDoF([bdNode;NO+bdNode;2*NO+bdNode;end]) = true;
freeDoF = find(~isBdDoF);

%% Initialize State at t=0

t = 0;
u0_node = pde.exact_u(node,0);
u0 = [u0_node(:); zeros(NT,1)];

sum_axp2 = 0;
sum_u_E2 = 0;

%% Time Loop

for it = 1:time_it
    t = t + dt;

    F = zeros(DoF,1);
    ft1 = zeros(NT,4); ft2 = zeros(NT,4); ft3 = zeros(NT,4);
    f0 = zeros(NT,1); pt = zeros(NT,1);

    if TestType == 1
        for p = 1:nQuad
            pxyz = lambda(p,1)*node(elem(:,1),:) + lambda(p,2)*node(elem(:,2),:) ...
                 + lambda(p,3)*node(elem(:,3),:) + lambda(p,4)*node(elem(:,4),:);
            fp = pde.rhs(pxyz,t);
            pp = pde.exact_p(pxyz,t);
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
    else % TestType == 2 or 3: pressure-robust body force via reconstruction
        ftL = zeros(NiF,1); ftR = zeros(NiF,1);

        for p = 1:nQuad
            pxyz = lambda(p,1)*node(elem(:,1),:) + lambda(p,2)*node(elem(:,2),:) ...
                 + lambda(p,3)*node(elem(:,3),:) + lambda(p,4)*node(elem(:,4),:);
            pxyzL = pxyz(TL,:); pxyzR = pxyz(TR,:);
            fp = pde.rhs(pxyz,t); fpL = pde.rhs(pxyzL,t); fpR = pde.rhs(pxyzR,t);
            pp = pde.exact_p(pxyz,t);
            PhiL = 2*(phi(p,i1L)'.*cross(DphiL2,DphiL3,2) + ...
                      phi(p,i2L)'.*cross(DphiL3,DphiL1,2) + ...
                      phi(p,i3L)'.*cross(DphiL1,DphiL2,2));
            PhiR = 2*(phi(p,i1R)'.*cross(DphiR2,DphiR3,2) + ...
                      phi(p,i2R)'.*cross(DphiR3,DphiR1,2) + ...
                      phi(p,i3R)'.*cross(DphiR1,DphiR2,2));
            ftL = ftL + weight(p)*dot(fpL,PhiL,2).*volume(TL);
            ftR = ftR + weight(p)*dot(fpR,PhiR,2).*volume(TR);
            for j = 1:4
                ft1(:,j) = ft1(:,j) + weight(p)*phi(p,j)*fp(:,1);
                ft2(:,j) = ft2(:,j) + weight(p)*phi(p,j)*fp(:,2);
                ft3(:,j) = ft3(:,j) + weight(p)*phi(p,j)*fp(:,3);
            end
            pt = pt + weight(p)*pp;
        end

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

    F(1:DoF_u) = F(1:DoF_u) + M*u0/dt;

    %% Apply Boundary Conditions and Solve

    x = zeros(DoF,1);
    uBd = pde.exact_u(node,t);
    x(bdNode)      = uBd(bdNode,1);
    x(NO+bdNode)   = uBd(bdNode,2);
    x(2*NO+bdNode) = uBd(bdNode,3);
    x(end)         = pt(end);

    F = F - A(:,isBdDoF)*x(isBdDoF);

    x(freeDoF) = A(freeDoF,freeDoF)\F(freeDoF);

    u0 = x(1:DoF_u);

    %% Compute Errors

    ph = x(3*NO+NT+1:3*NO+2*NT);
    uhCp = zeros(NO,3);
    uhCp(:,1) = x(1:NO); uhCp(:,2) = x(NO+1:2*NO); uhCp(:,3) = x(2*NO+1:3*NO);
    uhD = x(3*NO+1:3*NO+NT);

    Duh1 = Dphi(:,:,1).*uhCp(elem(:,1),1) + Dphi(:,:,2).*uhCp(elem(:,2),1) ...
         + Dphi(:,:,3).*uhCp(elem(:,3),1) + Dphi(:,:,4).*uhCp(elem(:,4),1) + uhD*[1 0 0];
    Duh2 = Dphi(:,:,1).*uhCp(elem(:,1),2) + Dphi(:,:,2).*uhCp(elem(:,2),2) ...
         + Dphi(:,:,3).*uhCp(elem(:,3),2) + Dphi(:,:,4).*uhCp(elem(:,4),2) + uhD*[0 1 0];
    Duh3 = Dphi(:,:,1).*uhCp(elem(:,1),3) + Dphi(:,:,2).*uhCp(elem(:,2),3) ...
         + Dphi(:,:,3).*uhCp(elem(:,3),3) + Dphi(:,:,4).*uhCp(elem(:,4),3) + uhD*[0 0 1];

    err_Dut = zeros(NT,1); err_ut = zeros(NT,1); err_pt = zeros(NT,1);
    for p = 1:nQuad
        pxyz = lambda(p,1)*node(elem(:,1),:) + lambda(p,2)*node(elem(:,2),:) ...
             + lambda(p,3)*node(elem(:,3),:) + lambda(p,4)*node(elem(:,4),:);
        uh = phi(p,1)*uhCp(elem(:,1),:) + phi(p,2)*uhCp(elem(:,2),:) ...
           + phi(p,3)*uhCp(elem(:,3),:) + phi(p,4)*uhCp(elem(:,4),:) + uhD.*(pxyz-xT);
        Du = pde.exact_gu(pxyz,t); uExa = pde.exact_u(pxyz,t); pval = pde.exact_p(pxyz,t);
        err_Dut = err_Dut + weight(p)*(sum((Duh1-Du(:,:,1)).^2,2) ...
                                     + sum((Duh2-Du(:,:,2)).^2,2) ...
                                     + sum((Duh3-Du(:,:,3)).^2,2));
        err_ut  = err_ut  + weight(p)*sum((uExa-uh).^2,2);
        err_pt  = err_pt  + weight(p)*(pval-ph).^2;
    end
    err_Dut = err_Dut.*volume; err_ut = err_ut.*volume; err_pt = err_pt.*volume;

    err_u0  = sqrt(sum(err_ut));
    err_u   = sqrt(sum(mu_elem.*err_Dut) + uhD'*B*uhD);
    err_axp = sqrt(sum((ph-pt).^2.*volume));
    err_p   = sqrt(sum(err_pt));
    sum_axp2 = sum_axp2 + err_axp^2;
    sum_u_E2 = sum_u_E2 + err_u^2;

end % time loop

err_axp_time = sqrt(dt * sum_axp2);
err_u_time = sqrt(dt * sum_u_E2);

fprintf('\n    ||u-u_h||_L2E : %.3e    ||u-u_h||_0 : %.3e    ||P_0p-p_h||_0 : %.3e    ||p-p_h||_0 : %.3e    ||P_0p-p_h||_L2L2 : %.3e\n\n', ...
    err_u_time, err_u0, err_axp, err_p, err_axp_time)
