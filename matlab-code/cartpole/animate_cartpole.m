function animate_cartpole(t,X,p, speed)
    if nargin < 4
        speed = 1;
    end
    t0 = t(1);
    time = t0;
    tic;
    pos = get_position_cartpole(X, p);
    xs = [pos(:, 1); pos(:, 3)];
    ys = [pos(:, 2); pos(:, 4)];
    margin = p.r * 0.5;
    extents = [min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin]; 
    while time < t(end)        
        time = t0 + (toc) * speed;
        posDraw = interp1(t, pos,time);
        draw(time,posDraw,extents);
        drawnow;
    end
    draw(t(end), pos(end, :), extents);
end


function draw(time, pos, extents)
    clf;
    hold on;
    [x1, y1, x2, y2] = deal(pos(1), pos(2), pos(3), pos(4));
    title(sprintf('t = %2.2f%',time));
    plot(extents(1:2),[0,0],'k-','LineWidth',2);
    plot(x1, y1, 'bs','MarkerSize',30,'LineWidth',5);
    plot([x1,x2], [y1, y2], 'r-','LineWidth',2);
    plot(x2, y2, 'ro','MarkerSize',22,'LineWidth',4);
    axis equal; axis(extents); axis off;
end
