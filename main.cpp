#include <iostream>
#include <vector>
#include "aos_equalization.h"
#include "soa_equalization.h"

int main() {
    const std::string inputFolder = "./Input";
    const std::string outputFolder = "./Output";
    const std::string imgExt = ".jpg";
    const std::vector<std::string> imgNames = {"4K_image", "Full_HD_image", "HD_image"};

    for (const auto &imgName:imgNames)
    {
        process_AoS_image_sequential(inputFolder+"/"+imgName+imgExt,
                            outputFolder+"/"+imgName+"_aos_seq"+imgExt);
        std::cout<<std::endl;

        process_SoA_image_sequential(inputFolder+"/"+imgName+imgExt,
                            outputFolder+"/"+imgName+"_soa_seq"+imgExt);
        std::cout<<std::endl;

        process_AoS_image_parallel(inputFolder+"/"+imgName+imgExt,
                            outputFolder+"/"+imgName+"_aos_par"+imgExt);
        std::cout<<std::endl;

        process_SoA_image_parallel(inputFolder+"/"+imgName+imgExt,
                            outputFolder+"/"+imgName+"_soa_par"+imgExt);
        std::cout<<std::endl;
    }

    return 0;
}
