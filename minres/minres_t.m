function [u,iter,resvec] = minres_t(A,b,M,u,tol,maxits,Test2Norm)
%MINRES_T My version of MINRES
%
% This is my version of MINRES -- adapted from the pdeminres file
%
% Input:  A :: matrix (can be matrix or function handle)
%         b :: rhs
%         M :: preconditioner (can be matrix or function handle)
%         u :: initial guess
%         tol :: tolerance
%         maxits :: max number of iterations
%         (Test2Norm) :: 1 - test the 2-norm
%                        0 - test the P^{-1} norm
%                       -1 - test both and print the difference
% Output: u :: solution
%         iter :: iteration number for convergence
%         resvec :: vector containing the residuals
%
% 
% T Rees 17.02.12
% T.R.   04.06.13 : allow A to be input as a function handle
%                   fix the description of inputs/outputs

if nargin == 6
    Test2Norm = 0;
end

resvec = zeros(maxits,1);
lb = length(b);

gam0 = 1;

v0 = sparse(lb,1);
w0 = sparse(lb,1); w1 = sparse(lb,1);
v1 = b-Amult(A,u);                                 % vold in matlab's code

z1 = presolve(M,v1);                        % u in matlab's code

gam1 = sqrt(z1'*v1);                        % beta1 in matlab's code
eta = gam1; s0 = 0; s1 = 0; c0 = 1; c1 = 1;
eta0 = eta;

if Test2Norm == 1 % test on the residual of 2 norm
    Aw0 = sparse(lb,1); Aw1 = sparse(lb,1);
    r = v1;
    test = norm(r);
elseif Test2Norm == -1 % do both
    test_2 = norm(v1);
    test = eta;
    diff = test_2 - test;
    fprintf('The difference in norms at iteration %i is %d\n',iter,diff)
else
    test = eta; % test on the the minres norm
end
test0 = abs(test);
resvec(1) = abs(test);
for iter = 1:maxits 
    z1 = z1/gam1;               % vv in matlab's code
    Az1 = Amult(A,z1);          % v in matlab's code
    del = Az1'*z1;              % alpha in matlab's code    
    v2 = Az1 - (del/gam1)*v1-(gam1/gam0)*v0; % matlab does the first part here, but not the second.  ML: v --> v2

    z2 = presolve(M,v2);

    gam2 = sqrt(z2'*v2);
    alp0 = c1*del-c0*s1*gam1;
    alp1 = sqrt(alp0^2+gam2^2);
    invse = 1/alp1;
    alp2 = s1*del + c0*c1*gam1;
    alp3 = s0*gam1;
    c0 = c1; s0 = s1; c1 = alp0*invse; s1 = gam2*invse;
    w2 = (z1 - alp3*w0-alp2*w1)*invse;
    if Test2Norm
        Aw2 = (Az1 - alp3*Aw0 - alp2*Aw1)*invse;
        r = r - c1*eta.*Aw2;
    end
    u = u + c1*eta.*w2;
    eta = -s1*eta;
    v0 = v1; v1=v2; z0=z1; z1=z2; w0=w1; w1=w2;  gam0 = gam1; gam1 ...
         = gam2;
    if Test2Norm == 1
        Aw0 = Aw1; Aw1 = Aw2;
        test = norm(r);
    elseif Test2Norm == -1 % do both
        test_2 = norm(b-Amult(A,u));
        test = eta;
        diff = test_2 - test;
        fprintf('The difference in norms at iteration %i is %d\n',iter,diff)        
    else
       test = eta;       % test on minres norm
    end
    
    resvec(iter+1) = abs(test);
    if abs(test)<tol*test0  
        %        fprintf('MINRES converged at iteration %i \n',iter);
        break
    end
    
end

resvec = resvec(1:iter+1);
if iter == maxits
    fprintf('Maximum number of iterations reached, MINRES failed to converge\n')
end
%%%%%%%%%%%%%%%
function x = presolve(P,b)
if isa(P,'function_handle')
    x = P(b);
else
    x = P\b;
end

%%%%%%%%%%%%%%%
function y = Amult(A,x)
if isa(A,'function_handle')
    y = P(x);
else
    y = A*x;
end