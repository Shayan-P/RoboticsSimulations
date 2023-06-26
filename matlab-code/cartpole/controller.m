function u = controller(trajectory, t, X)
    u = pd_controller(trajectory, t, X, 1, 0.1);
end

%% simple pd controller
function u = pd_controller(trajectory, t, X, pcof, dcof)
    index = max(find(trajectory.t <= t));
    ud = trajectory.u(index, :);
    Xd = trajectory.X(index, :)';
    qd = Xd(1:2);
    dqd = Xd(3:4);
    q = X(1:2);
    dq = X(3:4);
    e = qd - q; 
    de = dqd - dq;
    % u = ud + pcof * pinv(de/du) * e + dcof * pinv(dde/du) * de ??!
    % u = ud + pcof * sum(e) + dcof * sum(de);
    u = ud + pcof * e(1) + dcof * de(1);
end