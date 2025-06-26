clc, clearvars, close all;
function encrypted_image = image_encryption(input_image, seed_key, block_size, frft_order)   
    original_img = imread(input_image);
    if size(original_img, 3) == 3
        original_img = rgb2gray(original_img);
    end
    [h, w] = size(original_img);
    
    %image size to be divisible by block size
    padded_img = padarray(original_img, ...
                          [mod(block_size(1) - mod(h, block_size(1)), block_size(1)), ...
                           mod(block_size(2) - mod(w, block_size(2)), block_size(2))], ...
                          'post');
    [padded_h, padded_w] = size(padded_img);

    %Division into blocks
    rng(double(sum(seed_key)), 'twister');
    num_blocks_x = padded_w / block_size(2);
    num_blocks_y = padded_h / block_size(1);
    blocks = mat2cell(padded_img, repmat(block_size(1), 1, num_blocks_y), repmat(block_size(2), 1, num_blocks_x));

    %Shuffling blocks for transformed image
    total_blocks = num_blocks_x * num_blocks_y;
    shuffle_indices = randperm(total_blocks); % Random shuffle using seed
    transformed_blocks = blocks(shuffle_indices);
    transformed_image = cell2mat(reshape(transformed_blocks, num_blocks_y, num_blocks_x));
    
    %FRFT with random phase mask for each block
    encrypted_blocks = cell(size(transformed_blocks));
    for i = 1:total_blocks
        block = double(transformed_blocks{i});       
        phase_mask = exp(1i * 2 * pi * rand(size(block)));       
        block_with_mask = block .* phase_mask;
        
        %FRFT to the masked block
        encrypted_blocks{i} = abs(frft2d_custom(block_with_mask, frft_order)); % Magnitude only for display
    end
    encrypted_image = cell2mat(reshape(encrypted_blocks, num_blocks_y, num_blocks_x));    
    % Displaying original, transformed and encrypted images
    figure;
    subplot(1, 3, 1), imshow(original_img), title('Original Image');
    subplot(1, 3, 2), imshow(uint8(transformed_image), []), title('Transformed Image (Shuffled Blocks)');
    subplot(1, 3, 3), imshow(uint8(encrypted_image), []), title('Encrypted Image with FRFT');
end

% Helper Function: 2D FRFT for Image Blocks
function frft_block = frft2d_custom(block, alpha)
    % Apply FRFT to both rows and columns of the block using the provided frft function
    frft_block = arrayfun(@(i) frft(block(i, :), alpha), 1:size(block, 1), 'UniformOutput', false);
    frft_block = cell2mat(frft_block);
    frft_block = arrayfun(@(i) frft(frft_block(:, i), alpha), 1:size(frft_block, 2), 'UniformOutput', false);
    frft_block = cell2mat(frft_block);
end

%% Provided FRFT function
function Faf = frft(f, a)
    % The fast Fractional Fourier Transform
    % input: f = samples of the signal
    %        a = fractional power
    % output: Faf = fast Fractional Fourier transform

    f = f(:);
    N = length(f);
    shft = rem((0:N-1)+fix(N/2),N)+1;
    sN = sqrt(N);
    a = mod(a,4);
    % do special cases
    if (a==0), Faf = f; return; end
    if (a==2), Faf = flipud(f); return; end
    if (a==1), Faf(shft,1) = fft(f(shft))/sN; return; end 
    if (a==3), Faf(shft,1) = ifft(f(shft))*sN; return; end
    % reduce to interval 0.5 < a < 1.5
    if (a>2.0), a = a-2; f = flipud(f); end
    if (a>1.5), a = a-1; f(shft,1) = fft(f(shft))/sN; end
    if (a<0.5), a = a+1; f(shft,1) = ifft(f(shft))*sN; end
    % the general case for 0.5 < a < 1.5
    alpha = a*pi/2;
    tana2 = tan(alpha/2);
    sina = sin(alpha);
    f = [zeros(N-1,1) ; interp(f) ; zeros(N-1,1)];
    % chirp premultiplication
    chrp = exp(-1i*pi/N*tana2/4*(-2*N+2:2*N-2)'.^2);
    f = chrp.*f;
    % chirp convolution
    c = pi/N/sina/4;
    Faf = fconv(exp(1i*c*(-(4*N-4):4*N-4)'.^2),f);
    Faf = Faf(4*N-3:8*N-7)*sqrt(c/pi);
    % chirp post multiplication
    Faf = chrp.*Faf;
    % normalizing constant
    Faf = exp(-1i*(1-a)*pi/4)*Faf(N:2:end-N+1);
end

function xint = interp(x)
    % sinc interpolation
    N = length(x);
    y = zeros(2*N-1,1);
    y(1:2:2*N-1) = x;
    xint = fconv(y(1:2*N-1), sinc([-(2*N-3):(2*N-3)]'/2));
    xint = xint(2*N-2:end-2*N+3);
end

function z = fconv(x, y)
    % convolution by fft
    N = length([x(:); y(:)])-1;
    P = 2^nextpow2(N);
    z = ifft(fft(x, P) .* fft(y, P));
    z = z(1:N);
end
%%

% Parameters
input_image = 'rice.png';          
seed_key = 'encryptionKey123';     
block_size = [10, 10];              
frft_order = 0.5;                

% Encrypt the image
encrypted_image = image_encryption(input_image, seed_key, block_size, frft_order);
