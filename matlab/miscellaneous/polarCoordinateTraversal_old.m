function [radial_voxels, angular_voxels] = polarCoordinateTraversal_old(min_bound, max_bound, ray_origin, ray_direction, circle_center, ...
    circle_max_radius, num_radial_sections, num_angular_sections, t_begin, t_end, verbose)
% Input:
%    min_bound: The lower left corner of the bounding box.
%    max_bound: The upper right corner of the bounding box.
%    ray origin: The origin of the ray in (x, y) coordinates.
%    ray direction: The direction of the ray in (x, y) coordinates.
%    circle_center: The x, y location of the center of the circle.
%    circle_max_radius: The largest that encompasses the circle.
%    num_radial_sections: The number of radial sections in the circle.
%    num_angular_sections: The number of angular sections in the circle.
%    t_begin: The beginning time of the ray.
%    t_end: The end time of the ray.
%
% Requires:
%    max_bound > min_bound
%    circle center is within max_bound and min_bound.
%    t_end > t_begin >= 0.0
%    circle_max_radius > 0
%    num_radial_sections > 0
%    num_angular_sections > 0
%
% Returns:
%    radial_voxels: A list of the radial voxels that were hit by the ray.
%    angular_voxels: A list of the angular voxels that were hit by the ray.
%    These lists, used in conjunction, will produce the path of the ray
%    through the voxels using each point. For example,
%    [radial_voxels(1), angular_voxels(1)] is the first voxel the ray
%    travels through.
%
% Notes:
%    Currently under construction.
close all;
circle_center_x = circle_center(1);
circle_center_y = circle_center(2);
ray_origin_x = ray_origin(1);
ray_origin_y = ray_origin(2);
ray_direction_x = ray_direction(1);
ray_direction_y = ray_direction(2);

min_bound_x = min_bound(1);
min_bound_y = min_bound(2);
max_bound_x = max_bound(1);
max_bound_y = max_bound(2);

ray_start = ray_origin + t_begin * ray_direction;
ray_start_x = ray_start(1);
ray_start_y = ray_start(2);

ray_end = ray_origin + t_end * ray_direction;
ray_end_x = ray_end(1);
ray_end_y = ray_end(2);

if (verbose)
    figure;
    hold on;
    title('Polar Coordinate Voxel Traversal')
    
    if (t_begin ~= 0.0)
        % Mark the ray origin if the time does not start at 0.0
        text(ray_origin_x, ray_origin_y, ' ray origin');
        plot(ray_origin_x, ray_origin_y, 'k.', 'MarkerSize', 10);
        quiver(ray_origin_x, ray_origin_y, ray_direction_x, ray_direction_y, t_begin - 0.0, 'LineWidth', 1.5);
    end
    
    % Draw the ray.
    text(ray_start_x, ray_start_y, ' ray start');
    text(ray_end_x, ray_end_y, ' ray end');
    plot(ray_end_x, ray_end_y, 'k.', 'MarkerSize', 10);
    plot(ray_start_x, ray_start_y, 'k.', 'MarkerSize', 10);
    quiver(ray_start_x, ray_start_y, ray_direction_x, ray_direction_y, t_end - t_begin, 'LineWidth', 1.5);
    
    % Draw the axis.
    axis tight;
    xlim([min_bound_x, max_bound_x]);
    ylim([min_bound_y, max_bound_y]);
    xlabel('x');
    ylabel('y');
    grid on;
    
    % Draw the radial sections.
    current_max_radius = circle_max_radius;
    delta_radius = circle_max_radius / num_radial_sections;
    for k = 1:num_radial_sections
        viscircles(circle_center, current_max_radius, 'LineStyle', '--', 'Color', '#7E2F8E', 'LineWidth', 1);
        current_max_radius = current_max_radius - delta_radius;
    end
    
    % Draw the angular sections.
    N = num_angular_sections;
    section = 2 * pi / num_angular_sections;
    for ii = 1:N
        t = linspace(section * (ii - 1), section * (ii));
        x = circle_max_radius*cos(t) + circle_center_x;
        y = circle_max_radius*sin(t) + circle_center_y;
        x = [x circle_center_x x(1)];
        y = [y circle_center_y y(1)];
        line(x, y, 'LineStyle', '--', 'Color', '#7E2F8E', 'LineWidth', 0.5);
    end
end

% INITIALIZATION PHASE
% Does the ray intersect a radial shell?
% If yes, set Voxel ID R.
r = circle_max_radius;
ray_unit_vector = 1/sqrt(ray_direction(1)^2 + ray_direction(2)^2)...
    .* [ray_direction(1);  ray_direction(2)]';
ray_circle_vector = [circle_center(1) - ray_origin(1); circle_center(2) - ray_origin(2)]';
v = dot(ray_circle_vector,ray_unit_vector);
discr = r^2 - (dot(ray_circle_vector,ray_circle_vector) - v^2);
d = sqrt(discr);
p1 = ray_origin + (v-d).*ray_unit_vector
p2 = ray_origin + (v+d).*ray_unit_vector
t1 = (p1(1)-ray_origin(1))/ray_direction(1)
t2 = (p2(1)-ray_origin(2))/ray_direction(2)

if (t1 < 0 && t2 < 0 )
fprintf("\nray doesn't intersect polar grid!")
return;
end


while (ray_circle_vector(1)^2 + ray_circle_vector(2)^2 < r^2)
    r = r - delta_radius;        
end

if (t1 < 0) 
    t = t2;
    tStepR = -1;
    r = r + delta_radius;  
else
    t = t1;
    tStepR = 1; 
    r = r - delta_radius;  
end
p = ray_origin + t.*ray_unit_vector
if verbose
    if tStepR == 1
        text(p(1), p(2), 'POI_r');
        fprintf('RADIAL HIT (inward).\n');
    else
        text(p(1), p(2), 'POI_r');
        fprintf('RADIAL HIT (outward).\n');
    end
end

current_voxel_ID_r = (circle_max_radius - r)/delta_radius

pause
% II. Calculate Voxel ID Theta.
current_voxel_ID_theta = floor(atan2(ray_start_y - circle_center_y, ray_start_x - circle_center_x) * num_angular_sections / (2 * pi));
if current_voxel_ID_theta < 0
    current_voxel_ID_theta = num_angular_sections + current_voxel_ID_theta;
end
angular_voxels = [current_voxel_ID_theta];
radial_voxels = [current_voxel_ID_r];

% TRAVERSAL PHASE
while t < max(t1,t2)
    pause
    t
    % 1. Calculate tMaxR, tMaxTheta
    [tMaxR, tStepR] = radial_hit(ray_origin, ray_direction, ...
        current_voxel_ID_r, circle_center, circle_max_radius, delta_radius, r, t, verbose);
    pause
    
    [is_angular_hit, tMaxTheta, tStepTheta] = angular_hit(ray_origin, ray_direction, current_voxel_ID_theta,...
        num_angular_sections, circle_center, t, verbose);
    pause  
    tMaxTheta = inf
    % 2. Compare tMaxTheta, tMaxR
    if (tMaxTheta < tMaxR)
        t = tMaxTheta;
        current_voxel_ID_theta = current_voxel_ID_theta + tStepTheta;
        if verbose
            new_x_position = ray_origin(1) + ray_direction(1) * tMaxTheta;
            new_y_position = ray_origin(2) + ray_direction(2) * tMaxTheta;
            text(new_x_position, new_y_position, 'POI_t');
            fprintf('ANGULAR HIT.\n');
        end
    else
        t = tMaxR;
        current_voxel_ID_r = current_voxel_ID_r + tStepR;
        new_x_position = ray_origin_x + ray_direction_x * tMaxR;
        new_y_position = ray_origin_y + ray_direction_y * tMaxR;
        if (tStepR == 1)
          r = r - delta_radius;  
          elseif (tStepR == -1)
          r = r + delta_radius;
        else
          r = r;    
        end
        if verbose
            if tStepR == 1
                text(new_x_position, new_y_position, 'POI_r');
                fprintf('RADIAL HIT (inward).\n');
            else
                text(new_x_position, new_y_position, 'POI_r');
                fprintf('RADIAL HIT (outward).\n');
            end
        end
    end
    
    angular_voxels = [angular_voxels, current_voxel_ID_theta];
    radial_voxels = [radial_voxels, current_voxel_ID_r];
end


end
