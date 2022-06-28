# tampering_detection
The train dataset should be of the following structure

```bash
├───train 
    ├───Fake
    └───Original
```
### Requirement for the checked image
  
 :exclamation: **Image height and width must not be less than 64px**

### Installation guides
1. Install python 3.7+
2. Install packages from requirements.txt
3. Open detect.py and check if it is in debug or release mode (variable MODE can be 'DEBUG' or 'RELEASE')  
    For release mode - uncomment # MODE = 'RELEASE' and comment MODE = 'DEBUG'  
    To set parameters in the release mod, there are the following variables:  
    MODEL_PATH - path to saved model to be used for detection fake image  
    THRESHOLD - threshold value for making an image decision
    
4. Run detect.py

**Check image in debug mode:**
```bash
python detect.py --image-path <path/image_name> --create-map 1 --out-map-path <path> --threshold 0.02 --is-full-image 1
```
* `--image-path`: path to content image.
* `--image-path`: path to saved model 
* `--threshold`: threshold value for making an image decision.
* `--create-map`: set it to 1 for image output, 0 if not.
* `--out-map-path`: directory for save the image. (Example: C:\Project_files\images)
* `--is-full-image`: set it to 1 for checking the whole image, 0 if you want check local part of image with text

**Train model:** 
```bash
python cnn.py --epochs 50 --batch-size 32 --dataset <path/data_train> --save-model-dir <path/model_weight>
```
