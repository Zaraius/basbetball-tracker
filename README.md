# basbetball-tracker

We wanted to explore central machine vision topics of identifying an object and its position in 3D space, as well as determining it’s trajectory and estimating some kinematics of objects. We were hoping to frame this as analyzing a basketball as it is shot around the court, identifying its position, trajectory, velocity, and when it goes into a hoop.

Data collection:
We collected two videos and a series of photos for our data. All data was taken from an iPhone 12 Pro's camera, and a constant position and orientation between filming. We collected one 'real' video of us playing basketball and trying our best to mimic movements that would occur in a basketball game, i.e passing, dribbling, and shooting. This video was semi-random so we could ensure our scripts would work with a generic basketball game. You can tell the footage is real because we never made a single 3 point shot. This was at a series of unknown depths from the camera. We attempted to place Aruco tags for easy identification and real-world distances, but found that the camera couldn't accurately identify the tags, and ended up using ball size to estimate position instead. You can see the Aruco tags in our video, however.

Additionally, we filmed a video "side_by_side" to use for calibration of our ball tracking and kinematics scripts. As the title would suggest, the video shows Zaraius moving at a relatively constant speed moving right-to-left in the camera frame, holding the ball at a constant height and depth. This way, we can use the footage to validate our trajectory tracking by overlaying it over the video and looking for errors, and we can compare both the components and magnitude of velocity of our script. This was at a known depth from the camera.

Finally, we took a series of photos of the ball at a constant x and y position, but different z (depths). We measured each photo's depth and compared that to the size of the basketball to estimate a basketball size in pixels --> depth in feet transformation. Later on in this report, we use a Hough transformation to clearly identify the diameter of a basketball, and obtain a clearer depth. 

You may see some other students playing basketball in our videos. We told them ahead of time we would be filming and asked them not to play in our court, but unfortunately a few new students joined who were unaware. All students who appear in videos were told about them and allowed us to use the video, with the understanding we would not be tracking them. This provided us a with a few new challenges like what do we do when we identify multiple balls, but ultimately did not really affect our final project.


The first part of this requires us to identify the position of the ball in x and y(maybe z). We first need to find the ball in the camera frame. Our initial approach was to try a color mask to separate the ball and the background but we quickly realized this was an impossible task. The ball wouldn't always have consistent HSV values because of varying lighting and shadows. On top of this the floor and backdrop had similar hues as the ball.

This led us to try using an object identification model. We started with the yolo 11n object identification model and trained on a dataset of basketballs we found on the internet. 
https://universe.roboflow.com/eagle-eye/basketball-1zhpe/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true

After training this model we tested it out on a test video and noticed that it was good at generally finding the basketball, it missed a lot, especially when the ball was far away or slightly covered. We ran a script which generated 50 annotated images and manually fixed their annotations. We split these 50 images back into the train/test/validate(explain this? how yolo models work in general?) and retrained the yolo model. With this new model we are able to consistently get a bounding box for the ball. With the bounding box of the ball we can... segment? find its size??

Basics physics modeling - trajectory generation
We began by attempting to use Aruco tags in order to find values for real-world distances from the videos, but found it too difficult to capture the full court with our camera and have enough image resolution to identify tags. Instead we estimated distance by comparing ball diameter, as tracked by segmentation.py

Hough circle detection

Learnings/Conclusions
