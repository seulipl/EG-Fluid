function pde = Stokes_Time2

ProbType = 1;

pde = struct('exact_u',@ExaU,'exact_p',@ExaP,'rhs',@Rhs,'mu',@mu,...
    'exact_gu',@GradU,'time_factor',@time_factor);

    function tf_val = time_factor(t)
        switch ProbType
            case 1; tf_val = t^2;
            case 2; tf_val = sin(pi*t/2);
        end
    end

    function mu_val = mu(node)
        x = node(:,1);
        mu_val = (x-x+1)*1e-6;
    end

    function u = ExaU(node,t)
        x = node(:,1); y = node(:,2);
        switch ProbType
            case 1
                u(:,1) = (10*x.^2.*y.*(x-1).^2.*(2*y-1).*(y-1)).*t^2;
                u(:,2) = (-10*x.*y.^2.*(2*x-1).*(x-1).*(y-1).^2).*t^2;
            case 2
                u(:,1) =  sin(pi*x).*cos(pi*y)*sin(pi*t/2);
                u(:,2) = -cos(pi*x).*sin(pi*y)*sin(pi*t/2);
        end
    end

    function dt = dtU(node,t)
        x = node(:,1); y = node(:,2);
        switch ProbType
            case 1
                dt(:,1) = (10*x.^2.*y.*(x-1).^2.*(2*y-1).*(y-1))*(2*t);
                dt(:,2) = (-10*x.*y.^2.*(2*x-1).*(x-1).*(y-1).^2)*(2*t);
            case 2
                dt(:,1) =  sin(pi*x).*cos(pi*y)*(pi/2)*cos(pi*t/2);
                dt(:,2) = -cos(pi*x).*sin(pi*y)*(pi/2)*cos(pi*t/2);
        end
    end

    function Du = GradU(node,t)
        x = node(:,1); y = node(:,2);
        switch ProbType
            case 1
                Du(:,1,1) = (20*(x-1).*x.*(2*x-1).*(y-1).*y.*(2*y-1)).*t^2;
                Du(:,2,1) = (10*(x-1).^2.*x.^2.*(6*y.^2-6*y+1)).*t^2;
                Du(:,1,2) = (-10*(6*x.^2-6*x+1).*(y-1).^2.*y.^2).*t^2;
                Du(:,2,2) = (20*(x-1).*x.*(2*x-1).*(1-2*y).*(y-1).*y).*t^2;
            case 2
                Du(:,1,1) =  pi*cos(pi*x).*cos(pi*y)*sin(pi*t/2);
                Du(:,2,1) = -pi*sin(pi*x).*sin(pi*y)*sin(pi*t/2);
                Du(:,1,2) =  pi*sin(pi*x).*sin(pi*y)*sin(pi*t/2);
                Du(:,2,2) = -pi*cos(pi*x).*cos(pi*y)*sin(pi*t/2);
        end
    end

    function p = ExaP(node,t)
        x = node(:,1); y = node(:,2);
        switch ProbType
            case 1
                p = 10*(2*x-1).*(2*y-1)*t^2;
            case 2
                p = (sin(pi*x).*sin(pi*y) - 4/pi^2)*sin(pi*t/2);
        end
    end

    function f = Rhs(node,t)
        u = ExaU(node,t); dtu = dtU(node,t);
        x = node(:,1); y = node(:,2);
        mu_val = mu(node);
        switch ProbType
            case 1
                f(:,1) = dtu(:,1) + mu_val.*(-120*(y-1/2).*(x.^4-2*x.^3+(2*y.^2-2*y+1) ...
                    .*x.^2+(-2*y.^2+2*y).*x+y.^2/3-y/3)).*t^2 + 20*(2*y-1)*t^2;
                f(:,2) = dtu(:,2) + mu_val.*(240*(x-1/2).*((y.^2-y+1/6).*x.^2+ ...
                    (-y.^2+y-1/6).*x+y.^2.*(y-1).^2/2)).*t^2 + 20*(2*x-1)*t^2;
            case 2
                f(:,1) = sin(pi*x).*cos(pi*y)*(pi/2)*cos(pi*t/2) ...
                       + (2*mu_val*pi^2.*sin(pi*x).*cos(pi*y) + pi*cos(pi*x).*sin(pi*y))*sin(pi*t/2);
                f(:,2) = -cos(pi*x).*sin(pi*y)*(pi/2)*cos(pi*t/2) ...
                       + (-2*mu_val*pi^2.*cos(pi*x).*sin(pi*y) + pi*sin(pi*x).*cos(pi*y))*sin(pi*t/2);
        end
    end


end
