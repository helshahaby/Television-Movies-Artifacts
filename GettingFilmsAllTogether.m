tic

clc;
clear all;
close all;

workspace;
fontSize = 14;

% Terminate any opened word process
!taskkill -f -im WINWORD.EXE

Frame_ID = 0;
global InitialobjectRegion

InitialobjectRegion = [ 67    ,      11    ,    1139    ,     149 ];

% [x , y , width , height];
%upper left is [x,y] , lower right [width+x,height+y]
% Small Text
fineTuningRegion = [ -15 , -5 , 30 , 15];

% Large Text
% fineTuningRegion = [ -40 , -20 , 50 , 100];
spotsThreshold = 10;


%---------------------------------------------------------------------
% Done One Time

FilmAllTogther_dir = fullfile('E:\MATLAB_R2016a_Installation\MATLAB\MotionTextDetection\FilmAllTogther_dir');
FilmData = fullfile('E:\MATLAB_R2016a_Installation\MATLAB\MotionTextDetection\FilmData');

if ~exist(FilmAllTogther_dir , 'dir')
    mkdir FilmAllTogther_dir
end

if ~exist(FilmData , 'dir')
    mkdir FilmData
end


% % % % % copyfile GettingFilmsAllTogether.m FilmAllTogther_dir ;
copyfile VGG3_NET17b.mat FilmAllTogther_dir ;

% % % copyfile WonderWoman.avi FilmAllTogther_dir

%----------------------------------------------------------------------

% Save frames of the video
vidcap = VideoReader('E:\MATLAB_R2016a_Installation\MATLAB\sampled\Films\Train Videos\WonderWoman2017.mp4');
VideoFrameRate = vidcap.FrameRate;
ChannelName = 'Full_Film';
ProgramName = 'WonderWoman2017_VGG3';
typeOfSR = 'SplineSRInterpolation';

%-----------------------------------------------------------------------
cd('E:\MATLAB_R2016a_Installation\MATLAB\MotionTextDetection\FilmData')
 
% Done One Time
while hasFrame(vidcap)

    FrameRGB = readFrame(vidcap);

    %     %     Get the Desired part index for the rows
    %     if strcmp(MarkedPart , 'Bottom' ) == 1
    %         DesiredAreaIndex = round( size(FrameRGB,1) / 1.5);
    % %         DesiredWidthIndexStart = round( size(FrameRGB,2) / 4);
    % %         DesiredWidthIndexEnd = round( size(FrameRGB,2) * 3 / 4);
    % %         DesiredPartFrameRGB = FrameRGB(DesiredAreaIndex:end , DesiredWidthIndexStart:DesiredWidthIndexEnd , :) ;
    %         DesiredPartFrameRGB = FrameRGB(DesiredAreaIndex:end  , : , :) ;
    %     elseif strcmp(MarkedPart , 'Top' ) == 1
    %         DesiredAreaIndex = round(size(FrameRGB,1) / 3);
    %         DesiredPartFrameRGB = FrameRGB(1:DesiredAreaIndex , : , :) ;
    %     elseif strcmp(MarkedPart , 'Both' ) == 1
    %         DesiredPartFrameRGB = FrameRGB ;
    %     elseif strcmp(MarkedPart , 'Middle' ) == 1
    %     DesiredAreaIndexStart = round(size(FrameRGB,1) / 3);
    DesiredAreaIndexEnd   = round( size(FrameRGB,1)  / 1.5 );
    DesiredPartFrameRGB = FrameRGB(DesiredAreaIndexEnd : end, : , :) ;
    %     end
    %
    FrameGrayScale = rgb2gray(DesiredPartFrameRGB);


    %     FrameGrayScale = DesiredPartFrameRGB;

    compressedframe = imresize(FrameGrayScale, [ 40 , 200 ] );

    imwrite(compressedframe, [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d') , '.jpg']);

    Frame_ID = Frame_ID + 1 ;

    Index = Frame_ID + 1;

end

vidcap.delete();

%-------------------------------------------------------------------

cd ..
copyfile( 'E:\MATLAB_R2016a_Installation\MATLAB\MotionTextDetection\SRes' , FilmAllTogther_dir);
cd FilmAllTogther_dir

%-------------------------------------------------------------------
rng('default')

load VGG3_NET17b

net = VGG3_NET17b;

imds = imageDatastore('E:\MATLAB_R2016a_Installation\MATLAB\MotionTextDetection\FilmData');

YTestPred = classify( net , imds );

YTestPred = cellstr(YTestPred);
% copy files to the temp dir
% copyfile(fullfile(mlhdlc_demo_dir, [design_name,'.m*']), mlhdlc_temp_dir);

% % % save('WonderWoman2017_Le_YTestPred_Variable_Le_NET17b','YTestPred')
%--------------------------------------------------------------------

%Load "YTestPred" from "WonderWoman_YTestPred_Variable"
% % % load('WonderWoman2017_Le_YTestPred_Variable_Le_NET17b','YTestPred')

videoFileReader = vision.VideoFileReader('E:\MATLAB_R2016a_Installation\MATLAB\sampled\Films\Train Videos\WonderWoman2017.mp4');

Index = 1;
newFrame = [];
pastFrame = [];
LastTextFlag = 0;
FirstTextFlag = 0;
MiddleTextFlag = 0;
midFlag = 0;
i=0;

points = [];
points.Location = [];


isUpdatePointsValid = 0;
skipFrame = 0;
TextIsSameFlag = 1;
isLastFrameCornerCase2 = 0;
firstFrameFromMiddleOnesIsTakenFlag =0;
isNotLastFrameCornerCase1 = 0;

% Do not remove it even the previous Frame_ID variable exists because they
% are independent on each other
Frame_ID=0;


% Create document to save results
header2word(ChannelName , ProgramName , typeOfSR);

while ~isDone(videoFileReader)
    
    TextFlag = 0;   
      
    if LastTextFlag == 0 && skipFrame == 0 && isNotLastFrameCornerCase1 == 0
                
        %  Do not read the a new frame in the following cases
        newFrame = step(videoFileReader);
        %i = i+1  %  increased with every step used for debugging 
        DesiredAreaIndexEnd   = round( size(newFrame,1)  / 1.5 );
        DesiredPartnewFrame = newFrame(DesiredAreaIndexEnd : end, : , :) ;

        newFrame = rgb2gray(DesiredPartnewFrame);
        
    elseif LastTextFlag == 1
        
        pastFrame = newFrame;
        newFrame = nextFrame;
        
    end
    
    
    if (  (Index <= size(YTestPred,1)) && (strcmp(YTestPred(Index) , 'Text' ) == 1 ) )
        
        FirstTextFlag = 0;
        LastTextFlag = 0;
        TextFlag = 1;
        isNotLastFrameCornerCase1 = 0;
        
        
        % Get the first text frame
        if  (Index ~= 1 ) &&   strcmp(YTestPred(Index - 1) , 'Non_Text' ) == 1 &&    strcmp(YTestPred(Index + 1) , 'Text' ) == 1
            
            FirstTextFlag = 1;
            MiddleTextFlag = 1;
            movedPixelsFreeFrame = [];
            midFlag = 0;
            isUpdatePointsValid = 1;
            objectRegion = InitialobjectRegion ;
            
            FirstFrame_ID = Frame_ID;
            FirstTextFrame = newFrame -pastFrame;
            
            level = graythresh(FirstTextFrame);
            
            FirstTextFrame = imbinarize(FirstTextFrame,level);
            
            % Avoid tif files issue
            pastFrame = double(pastFrame);
            
            FirstTextFrame = double(FirstTextFrame);
                    
            % Write Frame Index minus 1 because I started from Frame_ID = 0
            imwrite(pastFrame, [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_BeforeFirst' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
            
            %imwrite(FirstTextFrame, [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_First' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
            
            %imwrite(~FirstTextFrame, [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_First_inverted' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
            
            pastFrame = newFrame;
            
        end
        
        % Get the Last text frame
        if (  Index < size(YTestPred , 1) && ( strcmp(YTestPred(Index + 1 ) , 'Non_Text' ) == 1 && strcmp(YTestPred(Index - 1 ) , 'Text' ) == 1 ) )
            
            nextFrame = step(videoFileReader);
            %i = i+1
            Frame_ID = Frame_ID + 1;
            Index = Frame_ID + 1;
            DesiredAreaIndexEnd   = round( size(nextFrame,1)  / 1.5 );
            DesiredPartnextFrame = nextFrame(DesiredAreaIndexEnd : end, : , :) ;
            
            nextFrame = rgb2gray(DesiredPartnextFrame);
            
            LastTextFrame =  newFrame - nextFrame;
            level = graythresh(LastTextFrame) ;
            
            LastTextFrame = imbinarize(LastTextFrame, level);
            
            LastTextFrameCropped= imcrop(LastTextFrame , objectRegion + fineTuningRegion );
            
            sizeCheck = size(LastTextFrameCropped , 1) == size(LastTextFrameCropped , 1 ) && size(LastTextFrameCropped , 2) == size(LastTextFrameCropped , 2) ...
                && size(image_thresholded , 1)== size(LastTextFrameCropped , 1 ) && size(image_thresholded , 2)== size(LastTextFrameCropped , 2); 
            
            middleLastCorrelation = corr2(image_thresholded , LastTextFrameCropped) ;
            
            if isLastFrameCornerCase2 ==0                
                firstLastCorrelation = corr2(FirstTextFrameCropped , LastTextFrameCropped);
            else
                firstLastCorrelation = corr2(FirstMiddleTextFrame , LastTextFrameCropped);
            end
            
            mitigateCase_1_error = firstLastCorrelation < 0.1;
			
            % Normal Case where Background Frames exist    
            if ( sizeCheck == 1 && ( middleLastCorrelation >= 0.3 ||  firstLastCorrelation >= 0.3))

                LastTextFlag = 1;
                MiddleTextFlag = 0;
                
                % in case that the true last frame is found , we can rest the flag
                isNotLastFrameCornerCase1 = 0;
                
                % Avoid tif files issue
                nextFrame = double(nextFrame);
                
                LastTextFrameCropped = double(LastTextFrameCropped);                
                
                imwrite(nextFrame, [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_Next' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                
                imwrite(LastTextFrameCropped, [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_Last' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                
                imwrite(~LastTextFrameCropped, [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_Last_inverted' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                
                SpotsFirstTextFrameCropped = (FirstTextFrameCropped == 1);
                SpotsFirstTextFrameCropped = imfill ( SpotsFirstTextFrameCropped , 'holes' );
                [~ , SpotsCountFirstTextFrameCropped] = bwlabel(SpotsFirstTextFrameCropped);
                
                Spotsimage_thresholded = (image_thresholded == 1);
                Spotsimage_thresholded = imfill ( Spotsimage_thresholded , 'holes' );
                [~ , SpotsCountimage_thresholded] = bwlabel(Spotsimage_thresholded);
                
                SpotsLastTextFrameCropped= (LastTextFrameCropped == 1);
                SpotsLastTextFrameCropped= imfill ( SpotsLastTextFrameCropped , 'holes' );
                [~ , SpotsCountLastTextFrameCropped] = bwlabel(SpotsLastTextFrameCropped);
                
                
                if SpotsCountFirstTextFrameCropped == 0 ||  mitigateCase_1_error == 1

                    % Mitigate Corner Case 1 error                    
                    ResultantTextFrame = image_thresholded | LastTextFrameCropped ;
                    

% % % % %                     spotsCount = [SpotsCountimage_thresholded SpotsCountLastTextFrameCropped];
% % % % %                     [Value , arrayIndex] = min(spotsCount);
% % % % %                     spotsCountDiff = [ SpotsCountimage_thresholded-spotsCount(arrayIndex) SpotsCountLastTextFrameCropped-spotsCount(arrayIndex)];
% % % % %                     spotsMask = spotsCountDiff < spotsThreshold;
% % % % %                     if isLastFrameCornerCase2 == 0
% % % % %                         if spotsMask(1) == 1
% % % % %                             ResultantTextFrame = ResultantTextFrame .* image_thresholded;
% % % % %                         end
% % % % %                         if spotsMask(2) == 1
% % % % %                             ResultantTextFrame = ResultantTextFrame  .* LastTextFrameCropped;
% % % % %                         end
% % % % %                     else
% % % % %                         if spotsMask(1) == 1
% % % % %                             ResultantTextFrame = image_thresholded;
% % % % %                         end
% % % % %                         if spotsMask(2) == 1
% % % % %                             ResultantTextFrame = ResultantTextFrame  .* LastTextFrameCropped;
% % % % %                         end
% % % % %                     end
                    
                else
                                        
                    if isLastFrameCornerCase2 == 0
					
                           % Normal Case
                           ResultantTextFrame = ( FirstTextFrameCropped + image_thresholded + LastTextFrameCropped ) >= 2;

% % % % %                         spotsCount = [SpotsCountFirstTextFrameCropped SpotsCountimage_thresholded SpotsCountLastTextFrameCropped];
% % % % %                         [Value , arrayIndex] = min(spotsCount);
% % % % %                         spotsCountDiff = [SpotsCountFirstTextFrameCropped-spotsCount(arrayIndex) SpotsCountimage_thresholded-spotsCount(arrayIndex) SpotsCountLastTextFrameCropped-spotsCount(arrayIndex)];
% % % % %                         spotsMask = spotsCountDiff < spotsThreshold;
% % % % % 					
% % % % %                         if spotsMask(1) == 1
% % % % %                             ResultantTextFrame = FirstTextFrameCropped;
% % % % %                         end
% % % % %                         if spotsMask(2) == 1
% % % % %                             ResultantTextFrame = ResultantTextFrame .* image_thresholded;
% % % % %                         end
% % % % %                         if spotsMask(3) == 1
% % % % %                             ResultantTextFrame = ResultantTextFrame  .* LastTextFrameCropped;
% % % % %                         end

						
                    else
                        % Corner case 2					
                        ResultantTextFrame = image_thresholded & LastTextFrameCropped;

% % % % %                         spotsCount = [SpotsCountimage_thresholded SpotsCountLastTextFrameCropped];
% % % % %                         [Value , arrayIndex] = min(spotsCount);
% % % % %                         spotsCountDiff = [SpotsCountimage_thresholded-spotsCount(arrayIndex) SpotsCountLastTextFrameCropped-spotsCount(arrayIndex)];
% % % % %                         spotsMask = spotsCountDiff < spotsThreshold;
% % % % % 										
% % % % %                         if spotsMask(1) == 1
% % % % %                             ResultantTextFrame = image_thresholded;
% % % % %                         end
% % % % %                         if spotsMask(2) == 1
% % % % %                             ResultantTextFrame = ResultantTextFrame  .* LastTextFrameCropped;
% % % % %                         end
% % % % %                         MiddleTextFlag = 1;
						
                    end
					
                end
								
                % Avoid tif files issue
                ResultantTextFrame = double(ResultantTextFrame);
                
                % Get the Resultant image from the first , middle and last frames
                
                imwrite(ResultantTextFrame, [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_Resultant' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                
                imwrite(~ResultantTextFrame, [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_Resultant_inverted' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                
                ResultantTextFrameInverted = ~ResultantTextFrame;
                
                FirstTextFrameCroppedInverted = ~FirstTextFrameCropped;
                
                image_thresholdedInverted = ~image_thresholded;
                
                LastTextFrameCroppedInverted = ~LastTextFrameCropped;
                
                clearvars gatheredForSR gatheredForSR_inverted
                 
                % Save all images in mat format
                
                gatheredForSR(:,:,1) = ResultantTextFrame;
% % %                 gatheredForSR(:,:,2) = image_thresholded;
% % %                 gatheredForSR(:,:,3) = LastTextFrameCropped;
                
                gatheredForSR_inverted(:,:,1) = ResultantTextFrameInverted;
% % %                 gatheredForSR_inverted(:,:,2) = image_thresholdedInverted;
% % %                 gatheredForSR_inverted(:,:,3) = LastTextFrameCroppedInverted;
                
% % %                 if isLastFrameCornerCase2 == 0              
% % %                     gatheredForSR(:,:,4) = FirstTextFrameCropped;
% % %                     gatheredForSR_inverted(:,:,4) = FirstTextFrameCroppedInverted;
% % %                 end
                                               
				% Reset Parameters that handle Corner case 2 condition
                firstFrameFromMiddleOnesIsTakenFlag = 0;
                isLastFrameCornerCase2 = 0;
                
% % %                 % Save all images in mat format
% % %                 save (strcat(ChannelName,'_',ProgramName,'_',num2str(FirstFrame_ID)) ,  'gatheredForSR' );
% % %                 
% % %                 save (strcat(ChannelName,'_',ProgramName,'_',num2str(FirstFrame_ID),'_Inverted') ,  'gatheredForSR_inverted');
                
                % Apply Super Resolution ( can remove output parasmeters )
                [HR , HR_Inverted ] = ApplySuperResolution ( gatheredForSR , gatheredForSR_inverted , ChannelName , ProgramName , FirstFrame_ID , typeOfSR);
                
                % Save image(s) to word
                save2word ( ChannelName , ProgramName , FirstFrame_ID , typeOfSR)              
                
                clearvars gatheredForSR gatheredForSR_inverted
                
            else
                % Make a correction [Corner Case 1]  
                % There is an error from the neural net Classification
%                 YTestPred(Index) = {'Text'};
%                 isNotLastFrameCornerCase1 = 1;
                %save('WonderWoman_YTestPred_Variable_Le_NET17b','YTestPred')
            end
        end
        
        
        if TextFlag ==  1 
            FirstTextFlag = 0 ;
            LastTextFlag = 0 ;
            if isNotLastFrameCornerCase1 == 0
                Frame_ID = Frame_ID + 1;
                Index = Frame_ID + 1;
            end
        end
        
        
        
        % Start         --------------------------------------------------------------------------------------------------------------------
             
        if isUpdatePointsValid == 1
            
            % No existence for [corner case 1]
            % Get next frame for point tracking
            nextFrame = step(videoFileReader);
            
            %i = i+1;
            
            DesiredAreaIndexEnd   = round( size(nextFrame,1)  / 1.5 );
            DesiredPartnextFrame = nextFrame(DesiredAreaIndexEnd : end, : , :) ;

            nextFrame = rgb2gray(DesiredPartnextFrame);

            nextFrameMovedFlag =  ~ (isequal(nextFrame , newFrame));
                        
                        
            % objectRegion = [x , y , width , height];
            % [x y] represent the location of the upper-left corner
            
            if (nextFrameMovedFlag)
                
                points = detectMinEigenFeatures( newFrame, 'ROI' , objectRegion);
                 
                % Update_ROI function will process the old points
                objectRegion = Update_ROI( newFrame , nextFrame , points ) ;
                
                %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% % %                 % Display Points
% % %                 newFrame = insertShape(newFrame,'Rectangle',objectRegion,'Color','red');
% % %                 figure;
% % %                 imshow(newFrame);
% % %                 title('Red box shows object region');
% % %                 %objectRegion=round(getPosition(imrect))
% % %                 
% % %                 
% % %                 pointImage = insertMarker( newFrame , points.Location ,'*' , 'Color' , 'green' );
% % %                 figure;
% % %                 imshow(pointImage);
% % %                 title('Detected interest points');
                %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            end
            
            isUpdatePointsValid = 0;
            
            skipFrame = 1;                    
            
        end
        

        if (TextFlag == 1 &&  skipFrame == 0)
           % Do Nothing
            
        elseif (TextFlag == 1 &&  skipFrame == 1 )
            
            TextFlag = 0;
            
            pastFrame = newFrame;
            
            newFrame = nextFrame;
            
            skipFrame = 0;   
            
        end
        
        textFrame = newFrame;        
        %imshow (textFrame);
       
        % End       -----------------------------------------------------------------------------------------------------
            
        if MiddleTextFlag == 1
            
            % Crop the text frame with margin 10 and 20 pixels
            textFrameCropped= imcrop(textFrame, objectRegion + fineTuningRegion );
 
            nextTextFrameCropped = imcrop(nextFrame, objectRegion + fineTuningRegion );
                        
            level_1 = 0.9;  % OR level_1 = graythresh(textFrameCropped);
            textFrameBinaryCropped = imbinarize(textFrameCropped , level_1);
            textFrameBinary = imbinarize(textFrame , level_1);
            
            if firstFrameFromMiddleOnesIsTakenFlag == 1
                
                FirstMiddleTextFrame = imcrop(pastTextFrameBinary, objectRegion + fineTuningRegion );
                RemoveMovedPixelsMask  = FirstMiddleTextFrame == textFrameBinaryCropped;
                movedPixelsFreeFrame = FirstMiddleTextFrame .* RemoveMovedPixelsMask;
                
            end
            
            level_2 = 0.9;  % OR level_2 = graythresh(nextTextFrameCropped);
            nextTextFrameBinaryCropped = imbinarize(nextTextFrameCropped , level_2);
            nextTextFrameBinary = imbinarize(nextFrame , level_2);
            
            RemoveMovedPixelsMask =  ( textFrameBinaryCropped == nextTextFrameBinaryCropped );
            
            % Multiply text frame by the mask
            if midFlag == 0
                
                if firstFrameFromMiddleOnesIsTakenFlag == 0
                    movedPixelsFreeFrame  =  textFrameBinaryCropped .* RemoveMovedPixelsMask ;
                end
                
                %To be used for Checking correlation each 1 sec
                FirstMiddleTextFrame = movedPixelsFreeFrame;
                
                midFlag = 1;
                
                % Normal Case to get the write the first frame with the same size of the middle ones
                if firstFrameFromMiddleOnesIsTakenFlag == 0
                    
                    % Save the First Frame
                    FirstTextFrameCropped= imcrop(FirstTextFrame , objectRegion + fineTuningRegion );
                    
                    % Avoid tif files issue
                    FirstTextFrameCropped = double(FirstTextFrameCropped);
                    
                    imwrite(FirstTextFrameCropped, [ChannelName,'_',ProgramName,'_',num2str(FirstFrame_ID, '%.6d'),'_First' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                    
                    imwrite(~FirstTextFrameCropped, [ChannelName,'_',ProgramName,'_',num2str(FirstFrame_ID, '%.6d'),'_First_inverted' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                end
                
            else
                
                % Check that the middle frame is still the same
                % In case that Corner Case 2 occurred then FirstTextFrameCropped is not equal FirstMiddleTextFrame
                
                corner2Corr_1 = corr2(FirstMiddleTextFrame , textFrameBinaryCropped);
                corner2Corr_2 = corr2(movedPixelsFreeFrame , textFrameBinaryCropped);
                
                TextIsSameFlag =  isnan(corner2Corr_1) == 1 ||  isnan(corner2Corr_2) == 1 || (corner2Corr_1 >= 0.4 ||  corner2Corr_2 >= 0.4) ;
                
                
                % Save First Frame ID to be used in case of Corner Case 2
                if TextIsSameFlag == 0 %&& firstFrameFromMiddleOnesIsTakenFlag == 1
                    FirstFrame_ID = Frame_ID + 1;
                end
                
                % In case that there is No background Frames [Corner Case 2]
                % i.e There is problem from the translation
                if TextIsSameFlag == 0
                    
                    TextIsSameFlag = 1;
                    
                    % Avoid tif files issue
                    nextFrame = double(nextFrame);
                    
                    % Take movedPixelsFreeFrame before the corruption
                    LastTextFrameCroppedFromMiddle = double(movedPixelsFreeFrame);
                    
                    % Save the Last Frame in case of {corner case 2]
                    if firstFrameFromMiddleOnesIsTakenFlag == 0
                        % Write real last Frame ID

                        imwrite(LastTextFrameCroppedFromMiddle, [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_CroppedFromMiddle_Last' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                        
                        imwrite(~LastTextFrameCroppedFromMiddle, [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_CroppedFromMiddle_Last_inverted' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                        
                    else
                        % Write the First and Last Middle Frames ID
                        imwrite(nextFrame, [ChannelName,'_',ProgramName,'_',num2str(FirstFrame_ID, '%.6d'),'_Next_ContainsText' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                        
                        imwrite(LastTextFrameCroppedFromMiddle, [ChannelName,'_',ProgramName,'_From_' , num2str(FirstFrame_ID, '%.6d') , '_To_' , num2str(Frame_ID, '%.6d'),'_CroppedFromMiddle_Last' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                        
                        imwrite(~LastTextFrameCroppedFromMiddle, [ChannelName,'_',ProgramName,'_From_' , num2str(FirstFrame_ID, '%.6d') , '_To_' , num2str(Frame_ID, '%.6d'),'_CroppedFromMiddle_Last_inverted' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                        
                    end
                    
                    if firstFrameFromMiddleOnesIsTakenFlag == 0
                        
                        ResultantTextFrame = FirstTextFrameCropped | LastTextFrameCroppedFromMiddle;
                        
% % % % %                         SpotsFirstTextFrameCropped = (FirstTextFrameCropped == 1);
% % % % %                         [~ , SpotsCountFirstTextFrameCropped] = bwlabel(SpotsFirstTextFrameCropped);
% % % % %                         
% % % % %                         SpotsLastTextFrameCroppedFromMiddle = (LastTextFrameCroppedFromMiddle == 1);
% % % % %                         [~ , SpotsCountLastTextFrameCroppedFromMiddle] = bwlabel(SpotsLastTextFrameCroppedFromMiddle);
% % % % %                         
% % % % %                         spotsCount = [SpotsCountFirstTextFrameCropped SpotsCountLastTextFrameCroppedFromMiddle];
% % % % %                         [Value , arrayIndex] = min(spotsCount);
% % % % %                         
% % % % %                         spotsCountDiff = [SpotsCountFirstTextFrameCropped-spotsCount(arrayIndex)  SpotsCountLastTextFrameCroppedFromMiddle-spotsCount(arrayIndex)];
% % % % %                         
% % % % %                         spotsMask = spotsCountDiff < spotsThreshold;
% % % % %                         
% % % % %                         if spotsMask(1) == 1
% % % % %                             
% % % % %                             ResultantTextFrame = FirstTextFrameCropped;
% % % % %                         else
% % % % %                             [m,n]=size(FirstTextFrameCropped);
% % % % %                             ResultantTextFrame = ones(m,n);
% % % % %                         end
% % % % %                         
% % % % %                         if spotsMask(2) == 1
% % % % %                             ResultantTextFrame = ResultantTextFrame  .* LastTextFrameCroppedFromMiddle;
% % % % %                         end

                        
                    else
                        ResultantTextFrame = LastTextFrameCroppedFromMiddle ;
                    end
                    
                    % Avoid tif files issue
                    ResultantTextFrame = double(ResultantTextFrame);
                    
                    %Get the Resultant image from the first (if exist ) and middle  frames  in case of {corner case 2]
                    
                    if firstFrameFromMiddleOnesIsTakenFlag == 0
                        % Write real last Frame ID
                        imwrite(ResultantTextFrame, [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_ResultantFromMiddle_Last' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                        
                        imwrite(~ResultantTextFrame, [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_ResultantFromMiddle_Last_inverted' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                        
                    else
                        % Write the First and Last Middle Frames ID
                        imwrite(ResultantTextFrame, [ChannelName,'_',ProgramName,'_From_' , num2str(FirstFrame_ID, '%.6d') , '_To_' , num2str(Frame_ID, '%.6d'),'_ResultantFromMiddle_Last' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                        
                        imwrite(~ResultantTextFrame, [ChannelName,'_',ProgramName,'_From_' , num2str(FirstFrame_ID, '%.6d') , '_To_' , num2str(Frame_ID, '%.6d'),'_ResultantFromMiddle_Last_inverted' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
                        
                    end
                    
                    ResultantTextFrameInverted = ~ResultantTextFrame;
                    LastTextFrameCroppedFromMiddleInverted = LastTextFrameCroppedFromMiddle;
                    
                    clearvars gatheredForSR gatheredForSR_inverted
                    
                    gatheredForSR(:,:,1) = ResultantTextFrame;
                    
                    gatheredForSR_inverted(:,:,1) = ResultantTextFrameInverted;
                    
% % %                     if firstFrameFromMiddleOnesIsTakenFlag == 0 %%&& spotsMask(1) == 1
% % %                         
% % %                         FirstTextFrameCroppedInverted = ~FirstTextFrameCropped;
% % %                         gatheredForSR(:,:,2) = FirstTextFrameCropped;
% % %                         gatheredForSR_inverted(:,:,2) = FirstTextFrameCroppedInverted;
% % %                         gatheredForSR(:,:,3) = LastTextFrameCroppedFromMiddle;
% % %                         gatheredForSR_inverted(:,:,3) = LastTextFrameCroppedFromMiddleInverted;
% % %                     end
                    
                    % Next Time the first frame will be taken from the middle
                    firstFrameFromMiddleOnesIsTakenFlag = 1;
                    
                    % Update the ROI Region due to new text frame in the middle
                    isUpdatePointsValid = 1;
                    
                    % Set flag for Last frame check
                    isLastFrameCornerCase2 = 1;
                    
% % %                     % Save all images in mat format
% % %                     save (strcat(ChannelName,'_',ProgramName,'_',num2str(FirstFrame_ID)) ,  'gatheredForSR' );
% % %                     
% % %                     save (strcat(ChannelName,'_',ProgramName,'_',num2str(FirstFrame_ID),'_Inverted') ,  'gatheredForSR_inverted');
% % %                     
                    % Apply Super Resolution
                    [HR , HR_Inverted ] = ApplySuperResolution ( gatheredForSR , gatheredForSR_inverted , ChannelName , ProgramName , FirstFrame_ID , typeOfSR);
                    
                    % Save image(s) to word
                    save2word ( ChannelName , ProgramName , FirstFrame_ID , typeOfSR);
                    
                else
                    
                    movedPixelsFreeFrame  = movedPixelsFreeFrame .* textFrameBinaryCropped .* RemoveMovedPixelsMask ;
                
                end
                
                
            end

            image_thresholded = movedPixelsFreeFrame;
            
            % Avoid tif files issue
            image_thresholded = double(image_thresholded);
            
            imwrite(image_thresholded, [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_Middle' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
            
            imwrite(~image_thresholded, [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_Middle_inverted' , '.tif']  , 'tif' , 'Resolution' , [300 , 300]  );
            
            pastTextFrameBinary = textFrameBinary;
            
        end
        
        
    end
    
    
    % Increase frame number in case that the frame is non text
    if TextFlag == 0
        Frame_ID = Frame_ID + 1;
        Index = Frame_ID + 1;
    end
    
    
    if LastTextFlag == 0
        pastFrame = newFrame;
    end
    
end

elapsedTime =  toc ;

Footer2word(ChannelName , ProgramName, typeOfSR , elapsedTime);

release(videoFileReader);
cd ..

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Implements a new object region detection
%
% Inputs:
% CroppednewFrame - Current Frame
% CroppednextFrame - New Frame
% points - are found using points tracker
%
% Output:
% optimized object Region
function objectRegion = Update_ROI(  CroppednewFrame , CroppednextFrame, points )

    global InitialobjectRegion;
    minROIflag = 0;
    maxROIflag = 0;
    
    if isempty ( points.Location )
        
        objectRegion = InitialobjectRegion;
        
    else
        
        tracker = vision.PointTracker('MaxBidirectionalError',1);
        
        initialize( tracker ,  points.Location , CroppednewFrame );
        
        %should be the next frame
        [points, validity] = step(tracker , CroppednextFrame);
        
        for i = 1 : length(points)
            if validity(i) == 1
                Row_min  =  points(i,1) ;
                %Col_min =   points(i,2) ;
                minROIflag = 1;
                break ;
            end
        end
        
        for i = 0:length(points)-1
            
            if validity(end-i) == 1
                Row_max =  points(end-i,1) ;
                %Col_max =  points(end-i,2) ;
                maxROIflag = 1;
                break;
            end
        end
        
        %objectRegion = [ x , y , width , height ] ;
        if minROIflag == 1 && maxROIflag == 1
            Col_min =   min(points(:,2)) ;
            Col_max =   max(points(:,2)) ;
            objectRegion = [  Row_min , Col_min  , Row_max - Row_min , abs(Col_max-Col_min ) ];
        else
            objectRegion = InitialobjectRegion ;
        end
        
    end

end




% Implements a simple cubic-spline interpolation of a single image. This
% image is then deblurred using the same method as in the Fast and Robust
% method.
%
% Inputs:
% LR - A sequence of low resolution images
%
% Output:
% The estimated HR image
function HR = SplineSRInterp( LR  )

    maxIter = 21;
    P = 2;
    alpha = 0.3;
    beta = 0.0001;
    lambda = 0.04;
    resFactor = 1;
    psfKernelSize = 2;
    psfSigma =  2;
   
    LR = double(LR);

    % Initialize guess as interpolated version of LR
    [X,Y] = meshgrid(0:resFactor:(size(LR,2)-1)*resFactor, 0:resFactor:(size(LR,1)-1)*resFactor);
    [XI,YI] = meshgrid(resFactor+1:(size(LR,2)-2)*resFactor-1, resFactor+1:(size(LR,1)-2)*resFactor-1);
    
    Z=interp2(X, Y, squeeze(LR(:,:,1)), XI, YI, '*spline');
    
    % Deblur the HR image and regulate using bilatural filter
    
    
    HR = Z;
    iter = 1;
    A = ones(size(HR));
    
    Hpsf = fspecial('gaussian', [psfKernelSize psfKernelSize], psfSigma);
    
    % Loop and improve HR in steepest descent direction
    while iter < maxIter
        
        % Compute gradient of the energy part of the cost function
        Gback = FastGradientBackProject(HR, Z, A, Hpsf);
        
        % Compute the gradient of the bilateral filter part of the cost function
        Greg = GradientRegulization(HR, P, alpha);
        
        % Perform a single SD step
        HR = HR - beta.*(Gback + lambda.* Greg);
        
        iter = iter+1;
        
    end
end





% Implements the fast and robust super-resolution method. This funtion
% first compute an estimation of the blurred HR image, using the median and
% shift method. It then uses the bilateral filter as a regulating term
% for the deblurring and interpolation step.
%
% Inputs:
% LR - A sequence of low resolution images
%
% Outputs:
% The estimated HR image
function HR = FastRobustSR( LR  )

    maxIter = 21;
    P = 2;
    alpha = 0.3;
    beta = 0.0001;
    lambda = 0.04;
    resFactor = 1;
    psfKernelSize = 2;
    psfSigma =  2;
    
    % Round translation to nearest neighbor
    D_Registeration = RegisterImageSeq(LR) ;
    D_Rounded = round( D_Registeration .*resFactor);
    % Shift all images so D is bounded from 0-resFactor
    D_Rounded=floor( D_Rounded /resFactor);
    D = mod(D_Rounded,resFactor)+resFactor;
    
    % Compute initial estimate of blurred HR by the means of MedianAndShift
    [Z, A] = MedianAndShift(LR, D, [(size(LR,1)+1)*resFactor-1 (size(LR,2)+1)*resFactor-1], resFactor);
    
    % Deblur the HR image and regulate using bilatural filter
    
    % Loop and improve HR in steepest descent direction
    HR = Z;
    iter = 1;    

    Hpsf = fspecial('gaussian', [psfKernelSize psfKernelSize], psfSigma);
    
    while iter < maxIter
        
        % Compute gradient of the energy part of the cost function
        Gback = FastGradientBackProject(HR, Z, A, Hpsf);
        
        % Compute the gradient of the bilateral filter part of the cost function
        Greg = GradientRegulization(HR, P, alpha);
        
        % Perform a single SD step
        HR = HR - beta.*(Gback + lambda.* Greg);
        
        iter = iter+1;
        
    end
end


% Implements the robust super-resolution method. This function uses the
% steepest descent method to minimize the SR cost function which includes
% two terms. The "energy" term, which is the L1 norm of the residual error
% between the HR image and the LR image sequence. The "regularization" term
% which induces piecewise smoothness on the HR image using the bilateral
% filter.
%
% Inputs:
% LR - A sequence of low resolution images
% InitialHR - InitialHR HR
% Output:
% The estimated HR image
function HR = RobustSR( LR , InitialHR )

    % Loop and improve HR in steepest descent direction
    iter = 1;
    
    maxIter = 21;
    P = 2;
    alpha = 0.7;
    beta = 0.1;
    lambda = 0.04;
    resFactor = 1;
    psfKernelSize = 3;
    psfSigma =  1;
    
    Hpsf = fspecial('gaussian', [psfKernelSize psfKernelSize], psfSigma);
    
    % Round translation to nearest neighbor
    D_Registeration = RegisterImageSeq(LR) ;
    D_Rounded = round( D_Registeration .*resFactor);
    % Shift all images so D is bounded from 0-resFactor
    D_Rounded=floor( D_Rounded /resFactor);
    D = mod(D_Rounded,resFactor)+resFactor;
    
    HR = InitialHR;
    
    while iter < maxIter
        
        % Compute gradient of the energy part of the cost function
        Gback = GradientBackProject(HR, LR, D, Hpsf, resFactor);
        
        % Compute the gradient of the bilateral filter part of the cost function
        Greg = GradientRegulization(HR, P, alpha);
        
        % Perform a single SD step
        HR = HR - beta.*(Gback + lambda.* Greg);
        
        iter = iter+1;
        
    end
end





% Implements a simple cubic-spline interpolation of a single image. This
% image is then deblurred using the same method as in the Fast and Robust
% method.
%
% Inputs:
% LR - A sequence of low resolution images
% LR_Inverted - LR Inverted
% ChannelName - Channel Name
% ProgramName - Program Name
% Frame_ID - Frame ID
% typeOfSR - type Of SR
%
% Output:
% The estimated HR image
function [HR , HR_Inverted]  = ApplySuperResolution( LR , LR_Inverted , ChannelName , ProgramName , Frame_ID , typeOfSR)


switch typeOfSR
    
    case 'SplineSRInterpolation'
        HR_SplineSRInterp = SplineSRInterp( LR );
        HR_SplineSRInterp_Inverted = SplineSRInterp( LR_Inverted );
        imwrite(HR_SplineSRInterp , [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_HR_SplineSRInterpolation' , '.tif']  , 'tif' , 'Resolution' , [300 , 300] );
        imwrite(HR_SplineSRInterp_Inverted , [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_HR_SplineSRInterpolation_Inverted' , '.tif']  , 'tif' , 'Resolution' , [300 , 300] );
        
        HR = HR_SplineSRInterp;
        HR_Inverted =  HR_SplineSRInterp_Inverted;
        
    case 'FastRobustSR'
        HR_FastRobustSR= FastRobustSR (LR );
        HR_FastRobustSR_Inverted= FastRobustSR (LR_Inverted );        
        imwrite(HR_FastRobustSR , [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_HR_FastRobustSR' , '.tif']  , 'tif' , 'Resolution' , [300 , 300] );
        imwrite(HR_FastRobustSR_Inverted , [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_HR_FastRobustSR_Inverted' , '.tif']  , 'tif' , 'Resolution' , [300 , 300] );
        
        HR = HR_FastRobustSR;
        HR_Inverted =  HR_FastRobustSR_Inverted;
        
    case 'RobustSR'
        HR_FastRobustSR= FastRobustSR (LR );
        HR_FastRobustSR_Inverted= FastRobustSR (LR_Inverted );        
        HR_RobustSR = RobustSR ( LR , HR_FastRobustSR);
        HR_RobustSR_Inverted = RobustSR ( LR_Inverted , HR_FastRobustSR_Inverted);        
        imwrite(HR_RobustSR , [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_HR_RobustSR' , '.tif']  , 'tif' , 'Resolution' , [300 , 300] );
        imwrite(HR_RobustSR_Inverted , [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_HR_RobustSR_Inverted' , '.tif']  , 'tif' , 'Resolution' , [300 , 300] );
        
        HR = HR_RobustSR;
        HR_Inverted =  HR_RobustSR_Inverted;
        
    otherwise
        HR_SplineSRInterp = SplineSRInterp( LR );
        HR_SplineSRInterp_Inverted = SplineSRInterp( LR_Inverted );
        imwrite(HR_SplineSRInterp , [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_HR_SplineSRInterpolation' , '.tif']  , 'tif' , 'Resolution' , [300 , 300] );
        imwrite(HR_SplineSRInterp_Inverted , [ChannelName,'_',ProgramName,'_',num2str(Frame_ID, '%.6d'),'_HR_SplineSRInterpolation_Inverted' , '.tif']  , 'tif' , 'Resolution' , [300 , 300] );
        
        HR = HR_SplineSRInterp;
        HR_Inverted =  HR_SplineSRInterp_Inverted;
        
end

end

% Implements the results header
%
% Inputs:
% ChannelName - Channel Name
% ProgramName - Program Name
% typeOfSR - type Of SR
%
% Output: -


function header2word(ChannelName , ProgramName , typeOfSR)

global word
global document
global selection

word = actxserver('Word.Application');      %start Word
% word.Visible =1;                          %make Word Visible
%                                           %for debugging

document=word.Documents.Add;                %create new Document
selection=word.Selection;                   %set Cursor
selection.Font.Name='Courier New';          %set Font
selection.Font.Size=14;                     %set Size

selection.Pagesetup.RightMargin=28.34646;   %set right Margin to 1cm
selection.Pagesetup.LeftMargin=28.34646;    %set left Margin to 1cm
selection.Pagesetup.TopMargin=28.34646;     %set top Margin to 1cm
selection.Pagesetup.BottomMargin=28.34646;  %set bottom Margin to 1cm
                                            %1cm is circa 28.34646 points
                                            
selection.Paragraphs.LineUnitAfter=0.01;    %sets the amount of spacing
                                            %between paragraphs(in gridlines)

selection.ParagraphFormat.Alignment =1;     %Center-aligned
selection.TypeText([ChannelName , ' ' , ProgramName] );
selection.TypeParagraph;                    %linebreak
selection.TypeParagraph;                    %linebreak
selection.ParagraphFormat.Alignment =0;     %Left-aligned

selection.TypeText(' This document is generated from Matlab for OCR purpose');
selection.TypeParagraph;                    %linebreak
selection.TypeParagraph;                    %linebreak
selection.TypeText([' Super Resolution is done using ' , typeOfSR ,' method']);
selection.TypeParagraph;                    %linebreak
selection.TypeParagraph;                    %linebreak
selection.TypeParagraph;                    %linebreak
selection.TypeParagraph;                    %linebreak

%save Document        
invoke(document,'SaveAs',[pwd , '/',ChannelName ,'_', ProgramName  ,'_', typeOfSR ,'.doc'],1);
end


% Implements the images saving in word
%
% Inputs:
% ChannelName - Channel Name
% ProgramName - Program Name
% Frame_ID - Frame ID
% typeOfSR - type Of SR
%
% Output: -

function save2word(ChannelName , ProgramName , Frame_ID , typeOfSR)

global word
global document
global selection

% Find end of document and make it the insertion point:
end_of_doc = get(word.activedocument.content,'end');
set( selection ,'Start',end_of_doc);
set( selection ,'End',end_of_doc);

selection.TypeText( [ 'Frame no. :' , num2str(Frame_ID+1, '%.6d')]);      %write number
selection.TypeParagraph;                    %linebreak
% selection.MoveUp(5,1,1);                    %5=row mode

selection.InlineShapes.AddPicture([pwd '/',ChannelName ,'_', ProgramName , '_',num2str(Frame_ID , '%.6d') ,'_HR_', typeOfSR,'.tif'],0,1);
selection.TypeParagraph;                    %linebreak
%with this command we insert a picture 'picture.png' wich is in the same
%folder as our m-file
selection.MoveDown(5,1);
selection.InlineShapes.AddPicture([pwd '/',ChannelName ,'_', ProgramName , '_', num2str(Frame_ID , '%.6d')  ,'_HR_', typeOfSR,'_Inverted' , '.tif'],0,1);
selection.TypeParagraph;                    %linebreak
selection.TypeParagraph;                    %linebreak
% selection.InsertNewPage;

%save Document        
invoke(document,'SaveAs',[pwd , '/',ChannelName ,'_', ProgramName  ,'_', typeOfSR ,'.doc'],1);
end


% Implements the results footer
%
% Inputs:
% ChannelName - Channel Name
% ProgramName - Program Name
% typeOfSR - type Of SR
% elapsedTime - Elapsed Time
%
% Output: -

function Footer2word( ChannelName , ProgramName, typeOfSR , elapsedTime)

global word
global document
global selection

% Find end of document and make it the insertion point:
end_of_doc = get(word.activedocument.content,'end');
set( selection ,'Start',end_of_doc);
set( selection ,'End',end_of_doc);

selection.TypeParagraph;                    %linebreak
selection.TypeParagraph;                    %linebreak
selection.TypeParagraph;                    %linebreak

selection.TypeText([ 'Total Elapsed Time : ' , num2str(elapsedTime)]);   %write number
selection.TypeParagraph;                    %linebreak
selection.MoveUp(5,1,1);                    %5=row mode

%save Document        
invoke(document,'SaveAs',[pwd , '/',ChannelName ,'_', ProgramName  ,'_', typeOfSR ,'.doc'],1);
word.Quit();                                %close Word

end

