num_of_images = 20;
list_of_images = cell(10,1);
j = '.jpg';
for i=1:num_of_images
    list_of_images{i} = [int2str(i) j];
end

images(num_of_images,1) = imFeature();
for i=1:num_of_images
    images(i).name = list_of_images{i};
    images(i).vec_feature = set_of_features(images(i));
end
X = zeros(num_of_images,14);
for i=1:num_of_images
    for j=1:14
        X(i,j)=images(i).vec_feature(j);
    end
end
IDX = kmeans(X,4);
kluster_1 = cell(20,1);
kluster_2 = cell(20,1);
kluster_3 = cell(20,1);
kluster_4 = cell(20,1);
for i=1:num_of_images
    if IDX(i)==1
        kluster_1{i}=list_of_images{i};
    elseif IDX(i)==2
        kluster_2{i}=list_of_images{i};
    elseif IDX(i)==4
        kluster_4{i}=list_of_images{i};
    else
        kluster_3{i}=list_of_images{i};
    end
end


