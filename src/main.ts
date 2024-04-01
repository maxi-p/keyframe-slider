
import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");

let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton: HTMLButtonElement;
let webcamRunning: Boolean = false;

// HandLandmarker class loading
const createHandLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: runningMode,
    numHands: 2
  });
  demosSection.classList.remove("invisible");
};
createHandLandmarker();


const video = document.getElementById("webcam") as HTMLVideoElement;
const canvasElement = document.getElementById(
  "output_canvas"
) as HTMLCanvasElement;
const canvasCtx = canvasElement.getContext("2d");


// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}
// Enable the live webcam view and start detection.
function enableCam(event) {
  if (!handLandmarker) {
    console.log("Wait! objectDetector not loaded yet.");
    return;
  }

  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE PREDICTIONS";
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE PREDICTIONS";
  }

  // getUsermedia parameters.
  const constraints = {
    video: true
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

let results = undefined;
console.log(video);
async function predictWebcam() {
  canvasElement.style.width = video.videoWidth;;
  canvasElement.style.height = video.videoHeight;
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  let cosine = 0;
  // Now let's start detecting the stream.
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await handLandmarker.setOptions({ runningMode: "VIDEO" });
  }
  let startTimeMs = performance.now();

  results = handLandmarker.detectForVideo(video, startTimeMs);
  // console.log(results)

  if(results.landmarks.length > 0){
    let zero = results.landmarks[0][0]
    let five = results.landmarks[0][5]
    let seventeen = results.landmarks[0][17]

    cosine = sliderValue(five.x ,five.y ,seventeen.x ,seventeen.y ,zero.x ,zero.y);
  }
  // console.log(results.landmarks)
  // console.log(sliderValue());

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  if (results.landmarks) {
    for (const landmarks of results.landmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: "#00FF00",
        lineWidth: 5
      });
      drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 2 });
    }
  }
  canvasCtx.font = "40px Arial";
  canvasCtx.strokeText(cosine.toFixed(1), 20, 100);
  canvasCtx.restore();

  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}

function sliderValue(a_x,a_y,b_x,b_y,c_x,c_y){
  // M - mid point between AB
  let m_x = (a_x+b_x)/2
  let m_y = (a_y+b_y)/2

  // vector CM
  let v_x = m_x-c_x;
  let v_y = m_y-c_y;
  let v_len = Math.sqrt(v_x*v_x+v_y*v_y);

  // vector AB
  let r_x = b_x-a_x;
  let r_y = b_y-a_y;
  let r_len = Math.sqrt(r_x*r_x+r_y*r_y);

  // magnitude of ABs projection on CM using Dot Product
  let cosine = (v_x*r_x + v_y*r_y)/(r_len*v_len);
  let sine = Math.sin(Math.acos(cosine));
  let h_len = r_len*sine*1000;
  
  // Sign of projection using Cross Product
  if((v_x*r_y-v_y*r_x)>0)
    return h_len;
  return -h_len;
}
