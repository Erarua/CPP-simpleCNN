#include"CNN.hpp"

int main(){

    std::string img_path = "./samples/face.jpg";
    //freopen("CNN_x86_out.txt","w",stdout);

    cv::Mat img_in = cv::imread(img_path,cv::IMREAD_UNCHANGED);

    //Check if the image has been read and continuous(in memory)
    //------------------------------------------------------------
    if( img_in.empty() ){

        std::fprintf( stderr, "No image input!\n" );
        return 0;

    }

    else{

        std::printf("Image has been read!\n");
        
        if( img_in.isContinuous() ){

            printf("Mat is continuous!\n");

        }

    }
    //------------------------------------------------------------

    
    clock_t start,end;


    Matrix in(img_in);


    start = clock();
    in.conv(0);
    end = clock();
    float total = 0.0f;
    total += 1000*difftime(end,start)/CLOCKS_PER_SEC;
    printf("Time used in conv_0: %.6fms\n",1000*difftime(end,start)/CLOCKS_PER_SEC);

    start = clock();
    in.maxpool();
    end = clock();
    total += 1000*difftime(end,start)/CLOCKS_PER_SEC;
    printf("Time used in max_pool_0: %.6fms\n",1000*difftime(end,start)/CLOCKS_PER_SEC);
    
    start = clock();
    in.conv(1);
    end = clock();
    total += 1000*difftime(end,start)/CLOCKS_PER_SEC;
    printf("Time used in conv_1: %.6fms\n",1000*difftime(end,start)/CLOCKS_PER_SEC);
    
    start = clock();
    in.maxpool();
    end = clock();
    total += 1000*difftime(end,start)/CLOCKS_PER_SEC;
    printf("Time used in max_pool_1: %.6fms\n",1000*difftime(end,start)/CLOCKS_PER_SEC);
    
    start = clock();
    in.conv(2);
    end = clock();
    total += 1000*difftime(end,start)/CLOCKS_PER_SEC;
    printf("Time used in conv_2: %.6fms\n",1000*difftime(end,start)/CLOCKS_PER_SEC);
    
    start = clock();
    in.flatten();
    end = clock();
    total += 1000*difftime(end,start)/CLOCKS_PER_SEC;
    printf("Time used in flatten: %.6fms\n",1000*difftime(end,start)/CLOCKS_PER_SEC);

    start = clock();
    in.fc();
    end = clock();
    total += 1000*difftime(end,start)/CLOCKS_PER_SEC;
    printf("Time used in full_connect: %.6fms\n",1000*difftime(end,start)/CLOCKS_PER_SEC);

    printf("Time used totally: %.6fms\n",total);
    
    in.softmax();


    printf("%dx%dx%d\n\n",in.channel,in.row,in.col);



    for(int c=0;c<in.channel;c++){
        for(int i=0;i<in.row;i++){
            for(int j=0;j<in.col;j++){

                printf("%.6f\n",in.data[i*in.col+j+c*in.row*in.col]);

            }
            // std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }

    
    fclose(stdout);


    return 0;
}