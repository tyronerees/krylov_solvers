function [x,iter,resvec] = preconjgrad(A,b,maxits,x,tol,P,verbose);

if nargin == 6
    verbose = 0;
end
% A plain preconditioned conjugate gradients routine
n = length(b);
resvec = zeros(maxits+1,1);

r = b - Amult(A,x);
z = presolve(P,r);
resvec(1) = norm(r);
p = z;
ztr = z'*r;

ANormEstimate = 0;
% Strakos and Tichy's lower bound on the A-norm of the error
% http://www.cs.cas.cz/tichy/download/public/2003StTiGamm.pdf

for iter = 1:maxits
    w = Amult(A,p);
    alpha = ztr/(w'*p);
    x = x + alpha*p;
    r_old = r;
    r = r_old - alpha*w;
    resvec(iter+1) = norm(r);
    ANormEstimate = ANormEstimate + alpha*ztr;
    if resvec(iter+1) < tol*resvec(1);
        if verbose, fprintf('converged!\n'); end
        return
    end
    z_old = z;
    z = presolve(P,r);
    ztr = z'*r;
    beta = (ztr)/(z_old'*r_old);
    p = z + beta*p;
end


%%%%
function y = Amult(A,x)
if isa(A,'function_handle')
    y = A(x);
else
    y = A*x;
end

%%%%
function x = presolve(P,b)
if isa(P,'function_handle')
    x = P(b);
else
    x = P\b;
end