function pde = Stokes_Time3

% ProbType = 1: manufactured solution on [0,1]^3
%   u = (sin(pi*x)cos(pi*y)-sin(pi*x)cos(pi*z), ...) * t^2
%   p = pi^3*sin(pi*x)*sin(pi*y)*sin(pi*z)
%
% ProbType = 2: channel flow past obstacle on (0,3/2)x(0,1)^2 \ K
%   f = 0, u = [1;0;0] on Gamma_in, u = 0 on Gamma_obs/Gamma_wall
%   do-nothing on Gamma_out, u0 = 0

ProbType = 1;

tf = @(t) t^2;
pde = struct('exact_u',@ExaU,'exact_p',@ExaP,'rhs',@Rhs,'mu',@mu,'exact_gu',@GradU,...
    'time_factor',tf,'inflow_u',@InflowU,'ProbType',ProbType);

    function mu_val = mu(node)
        x = node(:,1);
        mu_val = (x-x+1)*1e-3;
    end

    function u = ExaU(node,t)
        x = node(:,1); y = node(:,2); z = node(:,3);
        switch ProbType
            case 1
                u(:,1) = (sin(pi*x).*cos(pi*y) - sin(pi*x).*cos(pi*z))*t^2;
                u(:,2) = (sin(pi*y).*cos(pi*z) - sin(pi*y).*cos(pi*x))*t^2;
                u(:,3) = (sin(pi*z).*cos(pi*x) - sin(pi*z).*cos(pi*y))*t^2;
            case 2
                u = zeros(size(node,1),3);
        end
    end

    function u = InflowU(node)
        y = node(:,2); z = node(:,3);
        u = zeros(size(node,1),3);
        u(:,1) = 36*y.*(1-y).*z.*(1-z);  % parabolic profile, mean=1, zero on edges
    end

    function dtu = dtU(node,t)
        x = node(:,1); y = node(:,2); z = node(:,3);
        switch ProbType
            case 1
                dtu(:,1) = (sin(pi*x).*cos(pi*y) - sin(pi*x).*cos(pi*z))*(2*t);
                dtu(:,2) = (sin(pi*y).*cos(pi*z) - sin(pi*y).*cos(pi*x))*(2*t);
                dtu(:,3) = (sin(pi*z).*cos(pi*x) - sin(pi*z).*cos(pi*y))*(2*t);
            case 2
                dtu = zeros(size(node,1),3);
        end
    end

    function Du = GradU(node,t)
        x = node(:,1); y = node(:,2); z = node(:,3);
        switch ProbType
            case 1
                Du(:,1,1) = (pi*cos(pi*x).*cos(pi*y) - pi*cos(pi*x).*cos(pi*z))*t^2;
                Du(:,2,1) = (-pi*sin(pi*x).*sin(pi*y))*t^2;
                Du(:,3,1) = (pi*sin(pi*x).*sin(pi*z))*t^2;
                Du(:,1,2) = (pi*sin(pi*y).*sin(pi*x))*t^2;
                Du(:,2,2) = (pi*cos(pi*y).*cos(pi*z) - pi*cos(pi*y).*cos(pi*x))*t^2;
                Du(:,3,2) = (-pi*sin(pi*y).*sin(pi*z))*t^2;
                Du(:,1,3) = (-pi*sin(pi*z).*sin(pi*x))*t^2;
                Du(:,2,3) = (pi*sin(pi*z).*sin(pi*y))*t^2;
                Du(:,3,3) = (pi*cos(pi*z).*cos(pi*x) - pi*cos(pi*z).*cos(pi*y))*t^2;
            case 2
                Du = zeros(size(node,1),3,3);
        end
    end

    function p = ExaP(node,t)
        x = node(:,1); y = node(:,2); z = node(:,3);
        switch ProbType
            case 1
                p = pi^3*sin(pi*x).*sin(pi*y).*sin(pi*z) + t*0;
            case 2
                p = zeros(size(node,1),1);
        end
    end

    function f = Rhs(node,t)
        x = node(:,1); y = node(:,2); z = node(:,3);
        switch ProbType
            case 1
                mu_val = mu(node);
                g1 = sin(pi*x).*cos(pi*y) - sin(pi*x).*cos(pi*z);
                g2 = sin(pi*y).*cos(pi*z) - sin(pi*y).*cos(pi*x);
                g3 = sin(pi*z).*cos(pi*x) - sin(pi*z).*cos(pi*y);
                f(:,1) = g1*(2*t) + mu_val.*(2*pi^2*g1*t^2) + pi^4*cos(pi*x).*sin(pi*y).*sin(pi*z);
                f(:,2) = g2*(2*t) + mu_val.*(2*pi^2*g2*t^2) + pi^4*sin(pi*x).*cos(pi*y).*sin(pi*z);
                f(:,3) = g3*(2*t) + mu_val.*(2*pi^2*g3*t^2) + pi^4*sin(pi*x).*sin(pi*y).*cos(pi*z);
            case 2
                f = zeros(size(node,1),3);
        end
    end

end
