function pde = Brinkman2

ProbType = 1;

pde = struct('exact_u',@ExaU,'exact_p',@ExaP,'rhs',@Rhs,'nu',@nu,...
    'K',@K,'streamline',@streamline,'exact_gu',@GradU);

    function nu_val = nu(node)
        x = node(:,1);
        switch ProbType
            case 1
                nu_val = (x-x+1)*10^-6;
            case 2
                nu_val = x-x; % pure Darcy limit (no viscous term)
        end
    end

    function K_val = K(node)
        x = node(:,1);
        K_val = x-x+1;
    end

    function u = ExaU(node)
        x = node(:,1); y = node(:,2);
        switch ProbType
            case 1
                u(:,1) = 10*x.^2.*y.*(x-1).^2.*(2*y-1).*(y-1);
                u(:,2) = -10*x.*y.^2.*(2*x-1).*(x-1).*(y-1).^2;
            case 2
                u(:,1) = sin(pi*x).*sin(pi*y);
                u(:,2) = cos(pi*x).*cos(pi*y);
        end
    end

    function Du = GradU(node)
        x = node(:,1); y = node(:,2);
        switch ProbType
            case 1
                Du(:,1,1) = 20*(x-1).*x.*(2*x-1).*(y-1).*y.*(2*y-1);
                Du(:,2,1) = 10*(x-1).^2.*x.^2.*(6*y.^2-6*y+1);
                Du(:,1,2) = -10*(6*x.^2-6*x+1).*(y-1).^2.*y.^2;
                Du(:,2,2) = 20*(x-1).*x.*(2*x-1).*(1-2*y).*(y-1).*y;
            case 2
                Du(:,1,1) = pi*cos(pi*x).*sin(pi*y);
                Du(:,2,1) = pi*sin(pi*x).*cos(pi*y);
                Du(:,1,2) = -pi*sin(pi*x).*cos(pi*y);
                Du(:,2,2) = -pi*cos(pi*x).*sin(pi*y);
        end
    end

    function p = ExaP(node)
        x = node(:,1); y = node(:,2);
        switch ProbType
            case 1
                p = 10*(2*x-1).*(2*y-1);
            case 2
                p = sin(pi*x).*cos(pi*y);
        end
    end

    function f = Rhs(node)
        u = ExaU(node);
        x = node(:,1); y = node(:,2);
        u1 = u(:,1); u2 = u(:,2);
        nu_val = nu(node); K_val = K(node);
        switch ProbType
            case 1
                f(:,1) = nu_val.*(-120*(y-1/2).*(x.^4-2*x.^3+(2*y.^2-2*y+1) ...
                        .*x.^2+(-2*y.^2+2*y).*x+y.^2/3-y/3)) ...
                        + K_val.*u1 + 20*(2*y-1);
                f(:,2) = nu_val.*(240*(x-1/2).*((y.^2-y+1/6).*x.^2+ ...
                        (-y.^2+y-1/6).*x+y.^2.*(y-1).^2/2)) ...
                        + K_val.*u2 + 20*(2*x-1);
            case 2
                f(:,1) = nu_val*2*pi^2.*u1 + K_val.*u1 + pi*cos(pi*x).*cos(pi*y);
                f(:,2) = nu_val*2*pi^2.*u2 + K_val.*u2 - pi*sin(pi*x).*sin(pi*y);
        end
    end

    function u = streamline(x,y)
        u=5*x.^2.*(x-1).^2.*y.^2.*(y-1).^2;
    end

end
