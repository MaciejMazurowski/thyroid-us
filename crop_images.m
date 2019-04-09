function [ ] = crop_images( images_regex, cals_path, target_dim )
%CROP_IMAGES Crops nodules to square bounding box defined by callipers and
%resizes them to given dimensions.

if nargin < 1
    images_regex = '/data/images-cv/*.PNG';
end
if nargin < 2
    cals_path = '/data/detection/Calipers-cv';
end
if nargin < 3
    target_dim = 160;
end

margin = 32;

images_dir = dir(images_regex);

for i = 1:numel(images_dir)
    
    img_path = fullfile(images_dir(i).folder, images_dir(i).name);
    cal_filename = strrep(images_dir(i).name, 'PNG', 'csv');
    cal_path = fullfile(cals_path, cal_filename);
    
    cal = csvread(cal_path);
    cal = cal(:, 1:2);
    
    image = rgbread(img_path);
    
    image = medfilt3(image);
    
    image = crop2bbox(image, cal, margin);
    
    image = pad2square(image);
    
    image = imresize(image, [target_dim target_dim]);
    
    lims = [0.01 0.99];
    image = imadjust(image, stretchlim(image, lims), []);
    
    image = FastNonLocalMeans3D(double(image)/255, 0.1);
    image = uint8(image * 255);
    
    imwrite(image, img_path);
    
end

end


function [ padded ] = pad2square( img )

    if size(img, 1) == size(img, 2)
        padded = img;
        return;
    end
    
    if size(img, 1) < size(img, 2)
        ypad_post = ceil((size(img, 2) - size(img, 1)) / 2.0);
        ypad_pre = floor((size(img, 2) - size(img, 1)) / 2.0);
        xpad_post = 0;
        xpad_pre = 0;
    else
        xpad_post = ceil((size(img, 1) - size(img, 2)) / 2.0);
        xpad_pre = floor((size(img, 1) - size(img, 2)) / 2.0);
        ypad_post = 0;
        ypad_pre = 0;
    end
    
    padded = padarray(img, [ypad_post xpad_post], 0, 'post');
    padded = padarray(padded, [ypad_pre xpad_pre], 0, 'pre');

end

function [ cropped ] = crop2bbox( img, cals, margin )

    height = size(img, 1);
    width = size(img, 2);
    
    if size(cals, 1) <= 2
        center = [min(cals(:, 1)) + abs(cals(1, 1) - cals(2, 1)) / 2; ...
            min(cals(:, 2)) + abs(cals(1, 2) - cals(2, 2)) / 2];
        R = [cosd(90) -sind(90); sind(90) cosd(90)];
        cals = padarray(cals, [2 0], 1, 'post');
        cals(3, 1:2) = (R * (cals(1, 1:2)' - center) + center)';
        cals(4, 1:2) = (R * (cals(2, 1:2)' - center) + center)';
    end
    
    ymin = max(1, min(cals(:, 1)) - margin);
    ymax = min(max(cals(:, 1)) + margin, height);
    xmin = max(1, min(cals(:, 2)) - margin);
    xmax = min(max(cals(:, 2)) + margin, width);
    
    ymid = (ymax + ymin) / 2;
    xmid = (xmax + xmin) / 2;
    
    box_size = max((ymax - ymin), (xmax - xmin)) / 2;
    box_size = max(box_size, 80);
    
    ymin = round(max(1, ymid - box_size));
    ymax = round(min(ymid + box_size, height));
    xmin = round(max(1, xmid - box_size));
    xmax = round(min(xmid + box_size, width));
    
    cropped = img(ymin:ymax, xmin:xmax);

end

function [ rgb ] = rgbread( img_path )
%RGBREAD Reads image from given path and transforms it to RGB image if
%needed

[img, map] = imread(img_path);

if map
    rgb = ind2rgb(img, map);
else
    if size(img, 3) == 1
        rgb = cat(3, img, img, img);
    else
        rgb = img;
    end
end
