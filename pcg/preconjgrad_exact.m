function [x,iter,resvec] = preconjgrad_exact(A,b,maxits,x,tol,P,x_exact,verbose);

if nargin == 7
    verbose = 0;
end
% A plain preconditioned conjugate gradients routine
n = length(b);
resvec = zeros(maxits+1,1);

r = b - Amult(A,x);
z = presolve(P,r);
e = x - x_exact;
%resvec(1) = sqrt(e'*e);
resvec(1) = sqrt(e'*Amult(A,e));
p = z;

for iter = 1:maxits
    w = Amult(A,p);
    alpha = (z'*r)/(w'*p);
    x = x + alpha*p;
    r_old = r;
    r = r_old - alpha*w;
    e = x - x_exact;
    %    resvec(iter+1) = sqrt(e'*e);
    resvec(iter+1) = sqrt(e'*Amult(A,e));
    if resvec(iter+1) < tol%*resvec(1);
        if verbose 
            fprintf('converged @ iteration %d!\n',iter);
        end
        %        keyboard
        return
    end
    z_old = z;
    z = presolve(P,r);
    beta = (z'*r)/(z_old'*r_old);
    p = z + beta*p;
end


%%%%
function y = Amult(A,x)
if isa(A,'function_handle')
    y = P(x);
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