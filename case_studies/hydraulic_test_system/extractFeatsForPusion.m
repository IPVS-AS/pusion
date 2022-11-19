% Matlab script to export features from the time series data using the DAV3E toolbox.

clear all
run('..'); % add the path of the DAV3E toolbox

%% Read and fromat data

dataPath = '...'; % path where the data is located
profile=load([dataPath 'profile.txt']);

listSensors=dir([dataPath '*.txt']);
listSensors = listSensors(~ismember({listSensors.name},{'.','..','profile.txt'}));

for numSensor=1:size(listSensors,1)
    numSensor
    sensorNameTemp=listSensors(numSensor).name;
    sensorNames{numSensor}= erase(sensorNameTemp,'.txt');

    dataTemp=load([dataPath listSensors(numSensor).name]);
    dataTemp=dataTemp(profile(:,5)==0,:);
    trainData{numSensor}=dataTemp;
end 
    profile=profile(profile(:,5)==0,:);
    
    profileNew=sum(profile(:,2:4),2);
    unique(profileNew)
    
    
%% CLeSensors

idxToKeep=[4 5 6 11 13 16]; %% select flow, pressure and temperature sensors

trainDataNew = trainData(idxToKeep);
sensorNamesNew = sensorNames(idxToKeep);

  
%%
trainData=trainDataNew;
sensorNames=sensorNamesNew;
clc
extAlg = @StatisticalMoments;

extArg={}; %% Good results for 20,100
ext=MultisensorExtractor(extAlg, extArg);

ext.train(trainData);
featsAll=ext.apply(trainData);
infoFeats=ext.info();

%%
for numTarget=1:size(profile,2)-1
    
    sel=Pearson(3);
    sel.train(featsAll, profile(:,numTarget));
    featsSel{numTarget}=sel.apply(featsAll);
	idxFeatsSel{numTarget}=sel.rank(1:3);
    infoFeatsSel{numTarget}=infoFeats(idxFeatsSel{numTarget},:);
    
end
    idxFeatsSelVec=vertcat(idxFeatsSel{:});
    
    [FeatsSelVecUnique, idxUniqueFeats]=unique(idxFeatsSelVec,'stable');
    
    featsMultiClassUnique=featsAll(:,FeatsSelVecUnique);
    infoFeatsSelVecUnique=infoFeats(FeatsSelVecUnique,:);
    
    
    %%
save('featsForFusion4Sensors11Feats.mat', 'featsAll', 'featsMultiClassUnique', 'featsSel', 'infoFeatsSel', 'infoFeatsSelVecUnique',...
        'profile','sensorNames', 'trainData');
    
    
    
%% Analysis and evaluation with LMT DAV3E toolbox cross validation

% profile(:,5)=profile(:,1)+profile(:,4);
trainTarget=profile(:,2);

validation=cvpartition(trainTarget,'KFold', 10,'Stratify',true);


extractors = @StatisticalMoments;
extractorArgs={10};

cvPredictor = CrossValidator({@MultisensorExtractor, @Pearson,  @LDAMahalClassifier},{{extractors extractorArgs} {20} {}});  %Pearson Korrelation mit PLSR, 5 slektierte Merkmale 2 Komponenten
pred = cvPredictor.crossValidate(trainData, trainTarget, validation); 

e = ClassificationError.loss(pred, trainTarget);


figure(1)
plot(pred,'LineWidth',1)
hold on
plot(trainTarget,'LineWidth',1)





