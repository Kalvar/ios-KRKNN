//
//  ViewController.m
//  KRKNN
//
//  Created by Kalvar Lin on 2015/8/25.
//  Copyright (c) 2015年 Kalvar Lin. All rights reserved.
//

#import "ViewController.h"
#import "KRKNN.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad
{
    [super viewDidLoad];
    
    KRKNN *_knn = [KRKNN sharedInstance];
    
    // To use Cosine Simlarity or Euclidean that will have different results, suggests to use Cosine Similarity
    _knn.kernel = KRKNNKernelCosineSimilarity;
    //_knn.kernel = KRKNNKernelEuclidean;
    //_knn.kernel = KRKNNKernelRBF;
    
    // Features are wording appeared times on a paper as below like :
    // Apple, OS, Mobile, Taiwan, Japan, Developer
    [_knn addFeatures:@[@20, @9, @1, @3, @6, @2]
            groupName:@"Apple Fans"
           identifier:@"Smith"];
    
    [_knn addFeatures:@[@52, @32, @18, @7, @0, @1]
            groupName:@"Apple Fans"
           identifier:@"John"];
    
    [_knn addFeatures:@[@2, @20, @15, @5, @9, @16]
            groupName:@"Linux Fans"
           identifier:@"James"];
    
    [_knn addFeatures:@[@7, @11, @2, @12, @1, @0]
            groupName:@"Linux Fans"
           identifier:@"Terry"];
    
    [_knn addFeatures:@[@20, @8, @3, @21, @8, @25]
            groupName:@"Android Fans"
           identifier:@"Sam"];
    
    [_knn addFeatures:@[@2, @30, @8, @6, @33, @29]
            groupName:@"Android Fans"
           identifier:@"Amy"];
    
    // If you have batch-patterns (ex : 10 patterns) wanna classify that you could use for-loop to run the classify function,
    // In this demo that classifies a pattern by once time.
    [_knn classifyFeatures:@[@20, @1, @10, @2, @12, @3]
                identifier:@"Bob"
                 kNeighbor:[_knn chooseK]
                completion:^(BOOL success, NSString *ownGroup, NSInteger neighborCount, NSArray *allPatterns) {
                    if( success )
                    {
                        NSLog(@"ownGroup : %@", ownGroup);
                        NSLog(@"neighborCount : %li", neighborCount);
                        NSLog(@"allPatterns : %@", allPatterns);
                        // Looping that all classified patterns.
                        for( KRKNNPattern *pattern in allPatterns )
                        {
                            NSLog(@"pattern id is 「%@」 and group name is 「%@」", pattern.identifier, pattern.groupName);
                        }
                    }
                }];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    
}

@end
