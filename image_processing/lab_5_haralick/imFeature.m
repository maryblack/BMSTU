classdef imFeature
    properties
        name;
        vec_feature;
    end
    
    methods
        function obj = imFeature()
           
        end
    end
    methods 
        function x = set_of_features(im)
           S=imread(im.name);
           S=rgb2gray(S);
           I= imresize (S, [350 350]);
           glcm=graycomatrix(I,'offset',[-1 1],'NumLevel', 8,'Symmetric',true);
           xFeatures = 1:14;
           x = haralickTextureFeatures(glcm, xFeatures);
        end
    end
end

