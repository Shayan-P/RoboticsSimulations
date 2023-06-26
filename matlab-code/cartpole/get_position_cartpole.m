function pos = get_position_cartpole(X, p)
    x_ = X(:, 1);
    theta = X(:, 2);
    xc = x_;
    yc = zeros(size(xc));
    xe = xc + p.r * sin(theta);
    ye = yc + p.r * cos(theta);
    pos = [xc'; yc'; xe'; ye']';
end
