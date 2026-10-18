%% testscript for forward binding probability
% md ~ 26.04.2023

clear all;

D  = 5e-9*1e12;         % diffusion coefficient [m^2/s]
a  = .5e-6*1e6;         % receiver radius [m]
kf = 40*1e-14*1e18;     % forward reaction rate [m^3/s]
dt = .5e-7;             % temporal step size [s]

% simple forward probability [Mohammads boundary condition paper, kf in m/s]
kfc = kf/(4*pi*a^2);            % converting from m^3/s to m/s
prob_f1 = kfc * sqrt(pi*dt./D);

% precise calculation [Armans conference paper, kf in m^3/s?]
sigma = sqrt(4 * D * dt);
% FORMULA FROM ARMANS CONFERENCE PAPER (WRONG?!)
% prob_overlap = @(r) a./(2*sqrt(pi).*r) .*...
%     (exp(-(r+a).^2/sigma^2) - exp(-(r-a).^2/sigma^2)) +...
%     1/2*(erf((r+a)/sigma) + erf((a-r)/sigma));
% FORMULA FROM ORIGINAL CITED REFERENCE IN ARMAS CONFERENCE PAPER
prob_overlap = @(r) sigma./(2*sqrt(pi)*r) .*...
    (exp(-(r+a).^2/sigma^2) - exp(-(r-a).^2/sigma^2)) +...
    1/2*(erf((r+a)/sigma) + erf((a-r)/sigma));
rho = integral(@(r) r.^2 .* prob_overlap(r), a, inf);
prob_f2 = kf * dt / (4 * pi * rho);

% corrected kfc
kfc_correct = prob_f2./sqrt(pi*dt./D);
kfc_error = abs(kfc_correct-kfc)./kfc_correct;