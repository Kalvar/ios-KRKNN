Pod::Spec.new do |s|
  s.name         = "KRKNN"
  s.version      = "1.1.0"
  s.summary      = "k-Nearest Neighbor that classficiation method on Machine Learning."
  s.description  = <<-DESC
                   Machine Learning (マシンラーニング) in this project, it implemented KNN(k-Nearest Neighbor) that classification method. It can be used on products recommendation (おすすめの商品), user behavior analysis (ユーザーの行動分析), data mining (データマイニング) and data analysis (データ分析).
                   DESC
  s.homepage     = "https://github.com/Kalvar/ios-KRKNN"
  s.license      = { :type => 'MIT', :file => 'LICENSE' }
  s.author       = { "Kalvar Lin" => "ilovekalvar@gmail.com" }
  s.social_media_url = "https://twitter.com/ilovekalvar"
  s.source       = { :git => "https://github.com/Kalvar/ios-KRKNN.git", :tag => s.version.to_s }
  s.platform     = :ios, '7.0'
  s.requires_arc = true
  s.public_header_files = 'ML/**/*.h'
  s.source_files = 'ML/**/*.{h,m}'
  s.frameworks   = 'Foundation'
end
