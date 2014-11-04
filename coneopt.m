function [ x, s, y, z, output ] = coneopt( c, G, h, A, b, varargin )
% coneopt
% [ x, s, y, z, output ] = coneopt( c, G, h, A, b ) solves the cone LP (and dual)
%
%     minimize    c'*x              maximize    -h'*z - b'*y
%     subject to  G*x + s = h       subject to  G'*z + A'*y + c = 0
%                 A*x = b                       z in K.
%                 s in K
%
%   $Revision: 0.2.0 $  $Date: 2013/12/15 $
%
  REVISION = '$Revision: 0.2.0$';
  DATE     = '$Date: Dec. 15, 2013$';
  REVISION = REVISION(11:end-1);
  DATE     = DATE(8:end-1);

% ============ Process options ============

  default_options.debug      = 0;
  default_options.display    = 1;
  default_options.gap_tol    = 1e-6;
  default_options.lin_solver = 'chol';
  default_options.max_iter   = 100;
  default_options.step_len   = 0.99;
  default_options.rtol       = 1e-6;

  if nargin > 6
    options = merge_struct( default_options, varargin{1} );
  else
    options = default_options;
  end

  debug      = options.debug;
  display    = options.display;
  gap_tol    = options.gap_tol;
  lin_solver = options.lin_solver;
  max_iter   = options.max_iter;
  step_len   = options.step_len;
  rtol       = options.rtol;

% ============ Initialize ============

  [ m, ~ ] = size(G);
  [ p, n ] = size(A);

  e_K = ones(m,1);  m_K = m;

  FLAG_OPTIM   = 1;
  FLAG_PINF    = 2;
  FLAG_DINF    = 3;
  FLAG_MAXITER = 4;

  STATUS_OPTIM   = 'optimal';
  STATUS_PINF    = 'primal infeasible';
  STATUS_DINF    = 'dual infeasible';
  STATUS_MAXITER = 'max iterations reached';

% ------------ Initialize primal and dual variables ------------

  switch lin_solver
      case 'chol'
        [ L_1, L_2, chol_flag ] = kkt_chol( A, G );
        [ x_hat, y, s_tilde ]   = kkt_solve_chol( A, G, G, eye(m), L_1, L_2, chol_flag, ...
          zeros(n,1), b, h );
        [ x, y_hat, z_tilde ]   = kkt_solve_chol( A, G, G, eye(m), L_1, L_2, chol_flag, c, ...
          zeros(p,1), zeros(m,1) );
        s_tilde = - s_tilde;

      case 'qr'
        [ Q_1, Q_2, Q_3, R_1, R_3 ] = kkt_qr( A, G );
        [ x_hat, y, s_tilde ] = kkt_solve_qr( G, eye(m), Q_1, Q_2, Q_3, R_1, R_3, ...
          zeros(n,1), b, h );
        [ x, y_hat, z_tilde ] = kkt_solve_qr( G, eye(m), Q_1, Q_2, Q_3, R_1, R_3, ...
          c, zeros(p,1), zeros(m,1) );
        s_tilde = - s_tilde;

  end

  alpha_p = - min( s_tilde );
  alpha_d = - min( z_tilde );

  if alpha_p < 0
    s_hat = s_tilde;
  else
    s_hat = s_tilde + ( 1 + alpha_p ) * e_K;
  end

  if alpha_d < 0
    z_hat = z_tilde;
  else
    z_hat = z_tilde + ( 1 + alpha_d ) * e_K;
  end

  kappa_hat = 1;
  tau_hat   = 1;

% ------------ Compute Nesterov-Todd scaling and scaled variable ------------

  w      = sqrt( s_hat ) ./ sqrt( z_hat );  W = diag(w);

  lambda = sqrt( s_hat ) .* sqrt( z_hat );

  if display > 0
    fprintf( ' %s\n', repmat( '=', 1, 48 ) );
    fprintf( '         CONEOPT v.%s (%s)\n', REVISION, DATE );
    fprintf( ' %s\n', repmat( '=', 1, 48 ) );
    fprintf( ' %4s   %12s  %12s  %12s \n',...
      '','Primal res', 'Dual res.', 'Dual. gap' );
    fprintf( ' %s\n', repmat( '-', 1, 48 ) );
  end

% ============ Main loop ============

  iter = 0;

  trace.rel_pres = zeros( max_iter + 1, 1 );
  trace.rel_dres = zeros( max_iter + 1, 1 );
  trace.gap      = zeros( max_iter + 1, 1 );

  while 1

  % ------------ Compute residuals and gap ------------

    r_d   = A' * y_hat + G' * z_hat;
    r_x   = - r_d - c * tau_hat;
    r_p_y = A * x_hat;
    r_p_z = s_hat + G * x_hat;
    r_y   = r_p_y - b * tau_hat;
    r_z   = r_p_z - h * tau_hat;
    r_tau = kappa_hat + c' * x_hat + b' * y_hat + h' * z_hat;

    mu_hat = ( norm(lambda) ^ 2 + kappa_hat * tau_hat ) / ( m_K + 1 );

    rel_pres = norm(r_x) / max( 1, norm(c) ) / tau_hat;
    rel_dres = max( norm(r_y) / max( 1, norm(b) ), ...
                    norm(r_z) / max( 1, norm(h) ) ) ...
                    / tau_hat;
    gap = ( norm( lambda ) / tau_hat ) ^ 2;

    trace.rel_pres( iter + 1 ) = rel_pres;
    trace.rel_dres( iter + 1 ) = rel_dres;
    trace.gap( iter + 1 )      = gap;

    if display > 0
      fprintf( ' %4d | %12.4e  %12.4e  %12.4e\n', ...
        iter, rel_pres, rel_dres, gap );
    end

   % ------------ Check stopping condition ------------

    if ( rel_pres <= rtol && rel_dres <= rtol ) && gap <= gap_tol
      x = x_hat / tau_hat;  s = s_hat / tau_hat;
      y = y_hat / tau_hat;  z = z_hat / tau_hat;
      flag   = FLAG_OPTIM;
      status = STATUS_OPTIM;
      break

    elseif h' * z_hat + b' * y_hat < 0 && norm(r_d) / tau_hat <= rtol
      x = []   ;  s = [];
      y = y_hat;  z = z_hat;
      flag   = FLAG_PINF;
      status = STATUS_PINF;
      break

    elseif c' * x_hat < 0 && max( norm(r_p_y), norm(r_p_z) ) / tau_hat <= rtol
      x = x_hat;  s = z_hat;
      y = []   ;  z = [];
      flag   = FLAG_DINF;
      status = STATUS_DINF;
      break

    elseif iter >= max_iter
      x = x_hat / tau_hat;  s = s_hat / tau_hat;
      y = y_hat / tau_hat;  z = z_hat / tau_hat;
      flag   = FLAG_MAXITER;
      status = STATUS_MAXITER;
      break

    end

  % ------------ Factorize KKT system ------------

    G_tilde = W' \ G;

    switch lin_solver
      case 'chol'
        [ L_1, L_2, chol_flag ] = kkt_chol( A, G_tilde );

      case 'qr'
        [ Q_1, Q_2, Q_3, R_1, R_3 ] = kkt_qr( A, G_tilde );

    end

  % ------------ Compute affine search direction ------------

    d_s     = lambda .^ 2;
    d_kappa = kappa_hat * tau_hat;

    switch lin_solver
      case 'chol'
        [ x_1, y_1, W_z_1 ] = kkt_solve_chol( A, G, G_tilde, W, L_1, L_2, chol_flag, ...
          - c, b, h );
        [ x_2, y_2, W_z_2 ] = kkt_solve_chol( A, G, G_tilde, W, L_1, L_2, chol_flag, ...
          r_x, - r_y, s_hat - r_z );
        z_1 = W \ W_z_1;
        z_2 = W \ W_z_2;

      case 'qr'
        [ x_1, y_1, W_z_1 ] = kkt_solve_qr( G_tilde, W, Q_1, Q_2, Q_3, R_1, R_3, ...
          - c, b, h );
        [ x_2, y_2, W_z_2 ] = kkt_solve_qr( G_tilde, W, Q_1, Q_2, Q_3, R_1, R_3, ...
          r_x, - r_y, s_hat - r_z );
        z_1 = W \ W_z_1;
        z_2 = W \ W_z_2;

    end

    Delta_tau_a = ( r_tau - d_kappa / tau_hat + c' * x_2 + b' * y_2 + h' * z_2 ) / ...
      ( kappa_hat / tau_hat + norm( W_z_1 ) ^ 2 );
    Delta_x_a     = x_2 + Delta_tau_a * x_1;
    Delta_y_a     = y_2 + Delta_tau_a * y_1;
    Delta_z_a     = z_2 + Delta_tau_a * z_1;
    Delta_s_a     = - r_z - G * Delta_x_a + h * Delta_tau_a;
    Delta_kappa_a = - r_tau - c' * Delta_x_a - b' * Delta_y_a - h' * Delta_z_a;

    if debug
      res_aff = kkt_res( A, G, c, b, h, Delta_x_a, Delta_y_a, Delta_z_a, r_x, r_y, r_z, r_tau, ...
        lambda, W, Delta_s_a, d_s, kappa_hat, Delta_kappa_a, tau_hat, Delta_tau_a, d_kappa );

      if norm( res_aff ) > 1e-9
        warning( 'norm( res_aff ) = %12.4e', norm( res_aff ) )
      end

    end

  % ------------ Compute step length and centering parameter ------------

    Delta_s_a_tilde = W' \ Delta_s_a;
    Delta_z_a_tilde = W  * Delta_z_a;

    rho   = Delta_s_a_tilde ./ lambda;
    sigma = Delta_z_a_tilde ./ lambda;

    alpha_K   = 1 / max( [ 0; - min(rho); - min(sigma) ] );
    alpha_gap = 1 / max( [ 0; - Delta_kappa_a / kappa_hat; - Delta_tau_a / tau_hat ] );
    alpha     = min( [ 1; alpha_K; alpha_gap ] );

    sigma = ( 1 - alpha ) ^ 3;
    tau   = 1 - sigma;

  % ------------ Compute combined search direction ------------

    d_s     = lambda .^ 2 + ( Delta_s_a_tilde ) .* ( Delta_z_a_tilde ) - sigma * mu_hat * e_K;
    d_kappa = kappa_hat * tau_hat + Delta_kappa_a * Delta_tau_a - sigma * mu_hat;

    switch lin_solver
      case 'chol'
        [ x_1, y_1, W_z_1 ] = kkt_solve_chol( A, G, G_tilde, W, L_1, L_2, chol_flag, ...
          -c, b, h );
        [ x_2, y_2, W_z_2 ] = kkt_solve_chol( A, G, G_tilde, W, L_1, L_2, chol_flag, ...
          tau * r_x, - tau * r_y, W' * ( d_s ./ lambda ) - tau * r_z );
        z_1 = W \ W_z_1;
        z_2 = W \ W_z_2;

      case 'qr'
        [ x_1, y_1, W_z_1 ] = kkt_solve_qr( G_tilde, W, Q_1, Q_2, Q_3, R_1, R_3, ...
          - c, b, h );
        [ x_2, y_2, W_z_2 ] = kkt_solve_qr( G_tilde, W, Q_1, Q_2, Q_3, R_1, R_3, ...
          tau * r_x, - tau * r_y, W' * ( d_s ./ lambda ) - tau * r_z );
        z_1 = W \ W_z_1;
        z_2 = W \ W_z_2;

    end

    Delta_tau   = ( tau * r_tau - d_kappa / tau_hat + c' * x_2 + b' * y_2 + h' * z_2 ) / ...
      ( kappa_hat / tau_hat + norm( W_z_1 ) ^ 2 );
    Delta_x     = x_2 + Delta_tau * x_1;
    Delta_y     = y_2 + Delta_tau * y_1;
    Delta_z     = z_2 + Delta_tau * z_1;
    Delta_s     = - tau * r_z - G * Delta_x + h * Delta_tau;
    Delta_kappa = - tau * r_tau - c' * Delta_x - b' * Delta_y - h' * Delta_z;

    if debug
      res_com = kkt_res( A, G, c, b, h, Delta_x, Delta_y, Delta_z, tau * r_x, tau * r_y, tau * r_z, tau * r_tau, ...
        lambda, W, Delta_s, d_s, kappa_hat, Delta_kappa, tau_hat, Delta_tau, d_kappa );

      if norm( res_com ) > 1e-9
        warning( 'norm( res_com ) = %12.4e', norm( res_com ) )
      end

    end

  % ------------ Update iterates and scaling ------------

    Delta_s_tilde = W' \ Delta_s;
    Delta_z_tilde = W  * Delta_z;

    rho   = Delta_s_tilde ./ lambda;
    sigma = Delta_z_tilde ./ lambda;

    alpha_K   = 1 / max( [ 0; - min(rho); - min(sigma) ] );
    alpha_gap = 1 / max( [ 0; - Delta_kappa / kappa_hat; - Delta_tau / tau_hat ] );
    alpha     = step_len * min( [ 1; alpha_K; alpha_gap ] );

    x_hat = x_hat + alpha * Delta_x;  s_hat = s_hat + alpha * Delta_s;
    y_hat = y_hat + alpha * Delta_y;  z_hat = z_hat + alpha * Delta_z;
    kappa_hat = kappa_hat + alpha * Delta_kappa;
    tau_hat   = tau_hat   + alpha * Delta_tau;

    s_tilde = lambda + alpha * Delta_s_tilde;
    z_tilde = lambda + alpha * Delta_z_tilde;

    w      = sqrt( s_tilde ) ./ sqrt( z_tilde ) .* w;  W = diag( w );

    lambda = sqrt( s_tilde ) .* sqrt( z_tilde );

    iter = iter + 1;

  end

  trace.rel_pres = trace.rel_pres( 1 : iter + 1 );
  trace.rel_dres = trace.rel_dres( 1 : iter + 1 );
  trace.gap      = trace.gap( 1 : iter + 1 );

  output = struct( ...
    'flag'    , flag    , ...
    'iters'   , iter    , ...
    'options' , options , ...
    'status'  , status  , ...
    'trace'   , trace     ...
    );

  if display > 0
    fprintf( ' %s\n', repmat( '-', 1, 48 ) );
  end

end


function S3 = merge_struct( S1 ,S2 )
% merge_struct : merge two structures
%   self-explanatory ^
%
  S3 = S1;
  S3_names = fieldnames( S2 );
  for k = 1:length( S3_names )
    if isfield( S3, S3_names{k} )
      if isstruct( S3.(S3_names{k}) )
        S3.(S3_names{k}) = merge_struct( S3.(S3_names{k}),...
          S2.(S3_names{k}) );
      else
        S3.(S3_names{k}) = S2.(S3_names{k});
      end
    else
      S3.(S3_names{k}) = S2.(S3_names{k});
    end
  end
end


function [ L_1, L_2, chol_flag ] = kkt_chol( A, G_tilde )

  [ L_1, chol_flag ] = chol( G_tilde' * G_tilde, 'lower' );
  if chol_flag
    L_1 = chol( G_tilde' * G_tilde + A' * A, 'lower' );
  end

  A_tilde = L_1 \ A';
  L_2 = chol( A_tilde' * A_tilde, 'lower' );

end


function [ Q_1, Q_2, Q_3, R_1, R_3 ] = kkt_qr( A, G_tilde )

  [ p, n ] = size( A );

  [ Q, R ] = qr( A' );
    Q_1    = Q(:,1:p);  Q_2 = Q(:,p+1:n);
    R_1    = R(1:p,:);

  [ Q_3, R_3 ] = qr( G_tilde * Q_2, 0 );

end


function [ x, y, W_z ] = kkt_solve_chol( A, G, G_tilde, W, L_1, L_2, chol_flag, b_x, b_y, b_z )
% kkt_solve_chol : Solve KKT system with Cholesky factorization
%   Solve KKT systems of the form
%
%     [ 0  A'  G' ][ x ]   [ b_x ]
%     [ A  0   0  ][ y ] = [ b_y ] .
%     [ G  0  -Q  ][ z ]   [ b_z ]
%

  lin_solve_chol = @( L, b ) L' \ ( L \ b );

  if chol_flag
    y = lin_solve_chol( L_2, A * lin_solve_chol( L_1, b_x + G_tilde' * ( W' \ b_z ) + A' * b_y ) - b_y );
    x = lin_solve_chol( L_1, b_x + G_tilde' * ( W' \ b_z ) + A' * ( b_y - y ) );
  else
    y = lin_solve_chol( L_2, A * lin_solve_chol( L_1, b_x + G_tilde' * ( W' \ b_z ) ) - b_y );
    x = lin_solve_chol( L_1, b_x + G_tilde' * ( W' \ b_z ) - A' * y );
  end
  W_z = W' \ ( G * x - b_z );

end


function [ x, y, W_z ] = kkt_solve_qr( G_tilde, W, Q_1, Q_2, Q_3, R_1, R_3, b_x, b_y, b_z )
% kkt_solve_qr : Solve KKT system with QR factorization
%   Solve KKT systems of the form
%
%     [ 0  A'  G' ][ x ]   [ b_x ]
%     [ A  0   0  ][ y ] = [ b_y ] ,
%     [ G  0  -Q  ][ z ]   [ b_z ]
%
% where Q = W' * W, with a QR factorization.
%
  w   = W' \ b_z - G_tilde * Q_1 * ( R_1' \ b_y );
  u   = R_3' \ ( Q_2' * b_x ) + Q_3' * w;
  W_z = Q_3 * u - w;
  y   = R_1 \ ( Q_1' * ( b_x - G_tilde' * W_z ) );
  x   = Q_1 * ( R_1' \ b_y ) + Q_2 * ( R_3 \ u );

end


function res = kkt_res( A, G, c, b, h, Delta_x, Delta_y, Delta_z, r_x, r_y, r_z, r_tau, ...
    lambda, W, Delta_s, d_s, kappa, Delta_kappa, tau, Delta_tau, d_kappa )

  res_d_x     = - A' * Delta_y - G' * Delta_z - c * Delta_tau + r_x;
  res_d_y     = A * Delta_x - b * Delta_tau + r_y;
  res_d_z     = Delta_s + G * Delta_x - h * Delta_tau + r_z;
  res_d_tau   = Delta_kappa + c' * Delta_x + b' * Delta_y + h' * Delta_z + r_tau;
  res_d_s     = lambda .* ( W * Delta_z + W' \ Delta_s ) + d_s ;
  res_d_kappa = kappa * Delta_tau + tau * Delta_kappa + d_kappa;
  res         = [ res_d_x; res_d_y; res_d_z; res_d_tau; res_d_s; res_d_kappa ];

end

