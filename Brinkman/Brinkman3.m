function pde = Brinkman3

ProbType = 1;

pde = struct('exact_u',@ExaU,'exact_p',@ExaP,'rhs',@Rhs,'nu',@nu,...
    'K',@K,'streamline',@streamline,'exact_gu',@GradU);

    function nu_val = nu(node)
        x = node(:,1);
        nu_val = (x-x+1)*10^-6;
    end

    function K_val = K(node)
        x = node(:,1);
        K_val = x-x+1;
    end

    function u = ExaU(node)
        x = node(:,1); y = node(:,2); z = node(:,3);
        switch ProbType
            case 1
                u(:,1) = sin(pi*x).*cos(pi*y) - sin(pi*x).*cos(pi*z);
                u(:,2) = sin(pi*y).*cos(pi*z) - sin(pi*y).*cos(pi*x);
                u(:,3) = sin(pi*z).*cos(pi*x) - sin(pi*z).*cos(pi*y);
            case 2
                % constant unit flow (e.g. for a qualitative flow-around-obstacle test)
                u(:,1) = x-x+1;
                u(:,2) = y-y;
                u(:,3) = z-z;
        end
    end

    function Du = GradU(node)
        x = node(:,1); y = node(:,2); z = node(:,3);
        switch ProbType
            case 1
                Du(:,1,1) = pi*cos(pi*x).*cos(pi*y) - pi*cos(pi*x).*cos(pi*z);
                Du(:,2,1) = -pi*sin(pi*x).*sin(pi*y);
                Du(:,3,1) = pi*sin(pi*x).*sin(pi*z);
                Du(:,1,2) = pi*sin(pi*y).*sin(pi*x);
                Du(:,2,2) = pi*cos(pi*y).*cos(pi*z) - pi*cos(pi*y).*cos(pi*x);
                Du(:,3,2) = -pi*sin(pi*y).*sin(pi*z);
                Du(:,1,3) = -pi*sin(pi*z).*sin(pi*x);
                Du(:,2,3) = pi*sin(pi*z).*sin(pi*y);
                Du(:,3,3) = pi*cos(pi*z).*cos(pi*x) - pi*cos(pi*z).*cos(pi*y);
            case 2
                % u is constant, so its gradient is identically zero
                Du(:,1,1) = x-x; Du(:,2,1) = x-x; Du(:,3,1) = x-x;
                Du(:,1,2) = x-x; Du(:,2,2) = x-x; Du(:,3,2) = x-x;
                Du(:,1,3) = x-x; Du(:,2,3) = x-x; Du(:,3,3) = x-x;
        end
    end

    function p = ExaP(node)
        x = node(:,1); y = node(:,2); z = node(:,3);
        switch ProbType
            case 1
                p = pi^3*sin(pi*x).*sin(pi*y).*sin(pi*z)-1;
            case 2
                p = x-x+1;
        end
    end

    function f = Rhs(node)
        u = ExaU(node);
        x = node(:,1); y = node(:,2); z = node(:,3);
        u1 = u(:,1); u2 = u(:,2); u3 = u(:,3);
        K_val = K(node); nu_val = nu(node);
        switch ProbType
            case 1
                f(:,1) = pi*(2*pi*nu_val.*sin(pi*x).*cos(pi*y) - 2*pi*nu_val.*sin(pi*x).*cos(pi*z))...
                         + K_val.*u1 + pi^3*pi*cos(pi*x).*sin(pi*y).*sin(pi*z);
                f(:,2) = pi*(2*pi*nu_val.*sin(pi*y).*cos(pi*z) - 2*pi*nu_val.*sin(pi*y).*cos(pi*x))...
                         + K_val.*u2 + pi^3*pi*sin(pi*x).*cos(pi*y).*sin(pi*z);
                f(:,3) = pi*(2*pi*nu_val.*sin(pi*z).*cos(pi*x) - 2*pi*nu_val.*sin(pi*z).*cos(pi*y))...
                         + K_val.*u3 + pi^3*pi*sin(pi*x).*sin(pi*y).*cos(pi*z);
            case 2
                f(:,1) = x-x+1;
                f(:,2) = y-y+1;
                f(:,3) = z-z+1;
        end
    end

    function u = streamline(x,y)
        u=5*x.^2.*(x-1).^2.*y.^2.*(y-1).^2;
    end

end
