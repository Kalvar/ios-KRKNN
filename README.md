ios-KRKNN
=================

Machine Learning (マシンラーニング) in this project, it implemented KNN(k-Nearest Neighbor) that classification method. It can be used on products recommendation (おすすめの商品), user behavior analysis (ユーザーの行動分析), data mining (データマイニング) and data analysis (データ分析).

#### Podfile

```ruby
platform :ios, '7.0'
pod "KRKNN", "~> 1.1.4"
```

## How to use

#### To import the KRKNN.h file

``` objective-c
#import "KRKNN.h"
```

#### Normal Sample

Current implemented distance of kernel methods are :

``` objective-c
KRKNNKernelEuclidean
KRKNNKernelCosineSimilarity
KRKNNKernelRBF
```

``` objective-c
KRKNN *_knn = [KRKNN sharedInstance];
// To use Cosine Simlarity or Euclidean that will have different results, suggests to use Cosine Similarity
_knn.kernel = KRKNNKernelEuclidean;

// Features are the wordings appeared times on a paper as below :
// Apple, OS, Mobile, Taiwan, Japan, Developer
[_knn addFeatures:@[@20, @9, @1, @3, @6, @2]
            group:@"Apple Fans"
       identifier:@"Smith"];

[_knn addFeatures:@[@52, @32, @18, @7, @0, @1]
            group:@"Apple Fans"
       identifier:@"John"];

[_knn addFeatures:@[@2, @20, @15, @5, @9, @16]
            group:@"Linux Fans"
       identifier:@"James"];

[_knn addFeatures:@[@7, @11, @2, @12, @1, @0]
            group:@"Linux Fans"
       identifier:@"Terry"];

[_knn addFeatures:@[@20, @8, @3, @21, @8, @25]
            group:@"Android Fans"
       identifier:@"Sam"];

[_knn addFeatures:@[@2, @30, @8, @6, @33, @29]
            group:@"Android Fans"
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
```

## Version

V1.1.4

## License

MIT.
