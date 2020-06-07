function [train,test] = lvmLoadData(datafilename)

switch datafilename
%=================================================                   
    case 'food1'
              load('data\food1.mat');
              train = train1;
              test = test1;
              
    case 'food2'          
              load('data\food2.mat');
              train = train2;
              test = test2;
              
    case 'food3'          
              load('data\food3.mat');
              train = train3;
              test = test3;
              
    case 'food4'          
              load('data\food4.mat');
              train = train4;
              test = test4; 
              
    case 'food5'          
              load('data\food5.mat');
              train = train5;
              test = test5;           
%=================================================              
    case 'toy2_1'
               load('data\toy2_1.mat');
               train = train1;
               test = test1;
               
    case 'toy2_2'
               load('data\toy2_2.mat');
               train = train2;
               test = test2;  
               
    case 'toy2_3'
               load('data\toy2_3.mat');
               train = train3;
               test = test3;
    
    case 'toy2_4'
               load('data\toy2_4.mat');
               train = train4;
               test = test4;   
     
    case 'toy2_5'
               load('data\toy2_5.mat');
               train = train5;
               test = test5;  
%=================================================                                 
    case 'toy3_1'
               load('data\toy3_1.mat');
               train = train1;
               test = test1;
               
    case 'toy3_2'
               load('data\toy3_2.mat');
               train = train2;
               test = test2;  
               
    case 'toy3_3'
               load('data\toy3_3.mat');
               train = train3;
               test = test3;
    
    case 'toy3_4'
               load('data\toy3_4.mat');
               train = train4;
               test = test4;   
     
    case 'toy3_5'
               load('data\toy3_5.mat');
               train = train5;
               test = test5;      
%=================================================                                 
    case 'toy4_1'
               load('data\toy4_1.mat');
               train = train10;
               test = test10;
               
    case 'toy4_2'
               load('data\toy4_2.mat');
               train = train20;
               test = test20;  
               
    case 'toy4_3'
               load('data\toy4_3.mat');
               train = train30;
               test = test30;
    
    case 'toy4_4'
               load('data\toy4_4.mat');
               train = train40;
               test = test40;   
     
    case 'toy4_5'
               load('data\toy4_5.mat');
               train = train50;
               test = test50;                 
%=================================================                
    case 'sushi_1'
              load('data\sushi_1.mat');
              train = train1;
              test = test1;
              
    case 'sushi_2'
              load('data\sushi_2.mat');
              train = train2;
              test = test2;
              
    case 'sushi_3'          
              load('data\sushi_3.mat');
              train = train3;
              test = test3;
              
    case 'sushi_4'          
              load('data\sushi_4.mat');
              train = train4;
              test = test4;
              
    case 'sushi_5'          
              load('data\sushi_5.mat');
              train = train5;
              test = test5;              
%=================================================                
end



  
