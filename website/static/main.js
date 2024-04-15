import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';

import * as TWEEN from '@tweenjs/tween.js'

var default_data = {'li': [[0.3709, 0.3709, 0.3709, 0.3709, 0.3611],
  [0.341, 0.341, 0.341, 0.341, 0.3256]],
 'ul': [[-0.4672, -0.4672, -0.4672, -0.4672, -0.4838],
  [0.7996, 0.7996, 0.7996, 0.7996, 0.8175]],
 'll': [[0.2316, 0.2316, 0.2316, 0.2316, 0.2405],
  [-0.0309, -0.0309, -0.0309, -0.0309, -0.0322]],
 'tt': [[-0.561, -0.561, -0.561, -0.561, -0.5622],
  [0.9725, 0.9725, 0.9725, 0.9725, 0.99]],
 'tb': [[-0.8666, -0.8666, -0.8666, -0.8666, -0.8582],
  [0.2159, 0.2159, 0.2159, 0.2159, 0.231]],
 'td': [[-1.2006, -1.2006, -1.2006, -1.2006, -1.1923],
  [-0.0715, -0.0715, -0.0715, -0.0715, -0.0463]]};

var data = [{'li': [[0.3709, 0.3709, 0.3709, 0.3709, 0.3611],
  [0.341, 0.341, 0.341, 0.341, 0.3256]],
 'ul': [[-0.4672, -0.4672, -0.4672, -0.4672, -0.4838],
  [0.7996, 0.7996, 0.7996, 0.7996, 0.8175]],
 'll': [[0.2316, 0.2316, 0.2316, 0.2316, 0.2405],
  [-0.0309, -0.0309, -0.0309, -0.0309, -0.0322]],
 'tt': [[-0.561, -0.561, -0.561, -0.561, -0.5622],
  [0.9725, 0.9725, 0.9725, 0.9725, 0.99]],
 'tb': [[-0.8666, -0.8666, -0.8666, -0.8666, -0.8582],
  [0.2159, 0.2159, 0.2159, 0.2159, 0.231]],
 'td': [[-1.2006, -1.2006, -1.2006, -1.2006, -1.1923],
  [-0.0715, -0.0715, -0.0715, -0.0715, -0.0463]]}];

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
const renderer = new THREE.WebGLRenderer();
const controls = new OrbitControls( camera, renderer.domElement );
var avatar;

scene.add(new THREE.AxesHelper(5))
scene.fog = new THREE.Fog( 0x000001, 1, 300 );

const color2 = new THREE.Color(0x000000);

scene.background = color2;

const hemlight = new THREE.HemisphereLight( 0xffffbb, 0x080820, 1 );
scene.add (hemlight);

const outsideLight = new THREE.DirectionalLight( 0xffffff, 0.5 );

outsideLight.position.x = 0;
outsideLight.position.y = 70;
outsideLight.position.z = 100;

scene.add( outsideLight );
const amblight = new THREE.AmbientLight( 0x404040 ); // soft white light
scene.add( amblight );

//camera.position.set();
camera.position.z = 140;
camera.position.y = 10;

let width = window.innerWidth - 20;
let height = window.innerHeight - 200;

camera.aspect = width / height;
camera.updateProjectionMatrix();
renderer.setSize( width, height);
renderer.render( scene, camera );


let container = document.getElementById("avatar-container");
let status = document.getElementById("status");

container.appendChild( renderer.domElement );

renderer.domElement.id = "rt-avatar-canvas";

controls.target.y = 5;

const geometry = new THREE.BoxGeometry( 1, 1, 1 );
const material = new THREE.MeshBasicMaterial( { color: 0x00bb00 } );
const cube = new THREE.Mesh( geometry, material );
cube.position.z = 100;

var animId;

const loader = new FBXLoader()

function afterLoad(object) {
  createOffer();
  changeText();

  var i = 0;
  var total_frames = 0;
  var delay = 3;
  var parts_obj = getParts(object)
  var parts = ["tt", "tb", "td", "li", "ll", "ul"]
  var curr_data = data.pop();
  console.log(curr_data);
  var last_data = curr_data["tt"][0];

  // tweenUpdate(createTarget(curr_data, i), parts_obj);

  function animate() {
      if (total_frames % delay == 0) {
          tweenUpdate(createTarget(curr_data, i), parts_obj);
          console.log(i);
          if (i == curr_data["tt"][0].length - 1) {
              i = 0;
              if (data.length > 0) {
                  for (const part of parts) {
                      console.log(part);
                      default_data[part] = [
                          Array(1).fill(curr_data[part][0][4]),
                          Array(1).fill(curr_data[part][1][4])
                      ];
                  }
                  console.log(default_data);
                  curr_data = data.pop();
              } else {
                  curr_data = default_data;
              }
          } else {
              i += 1;
          }
      }
      total_frames += 1;


      renderer.render(scene, camera);
      animId = requestAnimationFrame( animate );
      //console.log(data);
      //controls.update();

      //console.log(TWEEN);
      //TWEEN.update(time);
      //renderer.render( scene, camera );
      //changeText();
  }
  animate();

  //ul.position.y = 0;
  //ul.position.z = 0;
  getParts(object);
  return object;

}


function tweenUpdate(position, parts) {
        //console.log(position);
        //console.log(parts.tt.position.y);
        //console.log(position.tty);
        parts.tt.position.y = position.tty;
        parts.tt.position.z = position.ttz;

        parts.tb.position.y = position.tby;
        parts.tb.position.z = position.tbz;

        parts.td.position.y = position.tdy;
        parts.td.position.z = position.tdz;

        parts.ul.position.y = position.uly;
        parts.ul.position.z = position.ulz;

        parts.ll.position.y = position.lly;
        parts.ll.position.z = position.llz;

        parts.li.position.y = position.liy;
        parts.li.position.z = position.liz;

        parts.li_hinge.position.z = position.li_hingez;

        parts.tongue_base.position.y = position.tongue_basey;
        parts.tongue_base.position.z = position.tongue_basez;

        parts.head_li.position.z = position.head_liz;

        parts.head_li_base.position.z = position.head_li_basez;

        parts.head_base.position.z = position.head_basez;

        parts.upper_teeth_joint.position.z = position.upper_teeth_jointz;
}

function createTarget(json_data, i) {
    return {
        tty: 3 * json_data["tt"][0][i] - 5, ttz: 3 * json_data["tt"][1][i] + 5,
        tby: 3 * json_data["tb"][0][i] - 5, tbz: 3 * json_data["tb"][1][i] + 5,
        tdy: 3 * json_data["td"][0][i] - 5, tdz: 3 * json_data["td"][1][i] + 5,
        uly: 2 * json_data["ul"][0][i] + 1, ulz: 3 * json_data["ul"][1][i] + 28,
        lly: 3 * json_data["ll"][0][i] - 16, llz: 3 * json_data["ll"][1][i] + 23,
        liy: 3 * json_data["li"][0][i] - 15, liz: 3 * json_data["li"][1][i] + 13,
        li_hingez: 3 * json_data["li"][1][i] - 0,
        tongue_basey: json_data["li"][0][i] - 0, tongue_basez: 0.7 * json_data["li"][1][i] - 10,
        head_liy: 3 * json_data["li"][0][i] - 20, head_liz: 3 * json_data["li"][1][i] + 17,
        head_li_basez: -9.563,
        head_basez: -5.343,
        upper_teeth_jointz: 3.985,
    }
}

function createPosition(object) {
    const parts = getParts(object)

    return {
        tty: parts.tt.position.y, ttz: parts.tt.position.z,
        tby: parts.tb.position.y, tbz: parts.tb.position.z,
        tdy: parts.td.position.y, tdz: parts.td.position.z,
        uly: parts.ul.position.y, ulz: parts.ul.position.z,
        lly: parts.ll.position.y, llz: parts.ll.position.z,
        liy: parts.li.position.y, liz: parts.li.position.z,
        li_hingez: parts.li_hinge.position.z,
        tongue_basey: parts.tongue_base.position.y, tongue_basez: parts.tongue_base.position.z,
        head_liy: parts.head_li.position.y, head_liz: parts.head_li.position.z,
        head_li_basey: parts.head_li_base.position.y, head_li_basez: parts.head_li_base.position.z,
        head_basey: parts.head_base.position.y, head_basez: parts.head_base.position.z,
        upper_teeth_jointy: parts.upper_teeth_joint.position.y, upper_teeth_jointz: parts.upper_teeth_joint.position.z
    }
}

function getParts(object) {
    return {
        head_base: object.children[1],
        upper_teeth_joint: object.children[2],
        li_hinge: object.children[3],
        tongue_base: object.children[4],
        head_li: object.children[6],
        ul: object.children[7],
        ll: object.children[8],
        head_li_base: object.children[9],
        li: object.children[10],
        tb: object.children[11],
        td: object.children[12],
        tt: object.children[13]
    }
}

let pc = new RTCPeerConnection();

async function createOffer() {
    console.log("Sending offer request");

    const offerResponse = await fetch("/offer", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            sdp: "",
            type: "offer",
        }),
    });

    const offer = await offerResponse.json();
    console.log("Received offer response:", offer);

    await pc.setRemoteDescription(new RTCSessionDescription(offer));

    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
}


async function changeText() {
    const response = await fetch("/audio_feed");
    const readableStream = response.body;
    const reader = readableStream.getReader();

    var audio = new Audio("static/wav/mngu0_s1_1165.wav");
    // audio.play();

    status.innerHTML = "listening...";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        console.log(done);
        var text = new TextDecoder("utf-8").decode(value);
        for (const part of split(text)) {
            console.log(part);
            var parsed_json = JSON.parse(part);
            data.push(parsed_json);
        }
    }
}

function split(data) {
    var parts = data.split("\n");
    parts = parts.filter(function (el) {
        return el !== "";
    });
    return parts;
}

function setupAvatar(object) {
  object.traverse(function (child) {
      if (child.isMesh) {
        // console.log(child.geometry.attributes.uv)
      }
  })

  object.scale.set(1, 1, 1)
  //object.computeFaceNormals();
  scene.add(object)

  var ignore1 = new THREE.Vector3();
  var ignore2 = new THREE.Vector3();
  var front_vector = new THREE.Vector3();

  // get the direction the camera is pointing at
  camera.matrix.extractBasis ( ignore1, ignore2, front_vector );

  // put the camera at a negative distance from the object
  //camera.position.copy(object.position);
  //camera.position.addScaledVector(front_vector, -4);

  //console.log(camera);
  //console.log(object)

  const faceMaterial = new THREE.MeshStandardMaterial();
  faceMaterial.polygonOffset = true;
  faceMaterial.polygonOffsetFactor = -0.1;
  const faceColor = new THREE.Color(0xd39972);
  faceMaterial.color = faceColor;

  const whiteMaterial = new THREE.MeshStandardMaterial();
  whiteMaterial.polygonOffset = true;
  whiteMaterial.polygonOffsetFactor = -0.1;
  const whiteColor = new THREE.Color('white');
  const blackColor = new THREE.Color(0x593716);
  const brownColor = new THREE.Color(0x9e6b4a);
  whiteMaterial.color = whiteColor;

  // Brown Irises
  object.children[1].children[1].material[1].color = brownColor;
  //console.log(object.children[1].children[1].material)

  // White Cornea
  object.children[1].children[1].material[0] = new THREE.MeshStandardMaterial();
  object.children[1].children[1].material[0].color = whiteColor;

  object.children[1].children[2].material[0] = object.children[1].children[1].material[0];

  // Face Fragment Color
  object.children[1].children[0].material[0].color = blackColor;
  object.children[1].children[0].material[1].color = faceColor;

  // Face Full Color
  //console.log(object.children[5].children[1])
  object.children[5].children[1].material = faceMaterial;
  //console.log(object.children[5].children[1])
  //object.children[5].children[1].material.color.r = 1;

  // Teeth
  //console.log(object.children[5].children)
  object.children[5].children[0].material = whiteMaterial;

  object.children[0].material = whiteMaterial;

  camera.translateZ(200);

  translateCamera();

}

let camera_position = 0;
function translateCamera() {
  camera.translateZ(-1);
  renderer.render(scene, camera);
  camera_position += 1;
  if (camera_position < 200) {
    requestAnimationFrame(translateCamera);
  }
}


loader.load( '/static/roger_avatar.fbx', (object) => {
  setupAvatar(object);
  afterLoad(object);
  renderer.render(scene, camera);
});

//console.log(avatar);
//animate();
