import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';

import * as TWEEN from '@tweenjs/tween.js'

var default_data = {'tt': [[-5.832118511199951,
   -5.109104633331299,
   -5.18584680557251,
   -5.314523696899414,
   -5.293256759643555],
  [9.198031425476074,
   9.513347625732422,
   9.402271270751953,
   9.440864562988281,
   9.3712797164917]],
 'tb': [[-5.51089334487915,
   -8.318242073059082,
   -8.411476135253906,
   -8.082327842712402,
   -7.7886786460876465],
  [3.0117759704589844,
   3.1890411376953125,
   2.7887284755706787,
   2.676365375518799,
   2.7128756046295166]],
 'td': [[-1.6218675374984741,
   -1.8297781944274902,
   -2.312278985977173,
   -2.425464153289795,
   -2.2355682849884033],
  [-9.245718955993652,
   -10.241634368896484,
   -10.25322151184082,
   -10.30083179473877,
   -10.327690124511719]],
 'ul': [[2.653531789779663,
   1.6134268045425415,
   1.5965423583984375,
   1.6634107828140259,
   1.8002780675888062],
  [22.548938751220703,
   22.55031394958496,
   22.708650588989258,
   22.71015167236328,
   22.66758155822754]],
 'll': [[-12.862114906311035,
   -11.87057113647461,
   -11.892444610595703,
   -12.273835182189941,
   -12.430438995361328],
  [25.569990158081055,
   25.18612289428711,
   25.46259307861328,
   25.52539825439453,
   25.504186630249023]],
 'li': [[-13.37097454071045,
   -13.528257369995117,
   -14.068887710571289,
   -14.300702095031738,
   -14.187480926513672],
  [18.33494758605957,
   18.791976928710938,
   18.829849243164062,
   18.842540740966797,
   18.8083438873291]]};

var data = [{'tt': [[-5.832118511199951,
   -5.109104633331299,
   -5.18584680557251,
   -5.314523696899414,
   -5.293256759643555],
  [9.198031425476074,
   9.513347625732422,
   9.402271270751953,
   9.440864562988281,
   9.3712797164917]],
 'tb': [[-5.51089334487915,
   -8.318242073059082,
   -8.411476135253906,
   -8.082327842712402,
   -7.7886786460876465],
  [3.0117759704589844,
   3.1890411376953125,
   2.7887284755706787,
   2.676365375518799,
   2.7128756046295166]],
 'td': [[-1.6218675374984741,
   -1.8297781944274902,
   -2.312278985977173,
   -2.425464153289795,
   -2.2355682849884033],
  [-9.245718955993652,
   -10.241634368896484,
   -10.25322151184082,
   -10.30083179473877,
   -10.327690124511719]],
 'ul': [[2.653531789779663,
   1.6134268045425415,
   1.5965423583984375,
   1.6634107828140259,
   1.8002780675888062],
  [22.548938751220703,
   22.55031394958496,
   22.708650588989258,
   22.71015167236328,
   22.66758155822754]],
 'll': [[-12.862114906311035,
   -11.87057113647461,
   -11.892444610595703,
   -12.273835182189941,
   -12.430438995361328],
  [25.569990158081055,
   25.18612289428711,
   25.46259307861328,
   25.52539825439453,
   25.504186630249023]],
 'li': [[-13.37097454071045,
   -13.528257369995117,
   -14.068887710571289,
   -14.300702095031738,
   -14.187480926513672],
  [18.33494758605957,
   18.791976928710938,
   18.829849243164062,
   18.842540740966797,
   18.8083438873291]]}]

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
camera.position.y = 0;

renderer.setSize( window.innerWidth - 10, window.innerHeight );
document.body.appendChild( renderer.domElement );

controls.target.y = 5;

const geometry = new THREE.BoxGeometry( 1, 1, 1 );
const material = new THREE.MeshBasicMaterial( { color: 0x00bb00 } );
const cube = new THREE.Mesh( geometry, material );
const clock = new THREE.Clock();
const speed = 0.4167;
var delta = 0;
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
              if (i > 10) {
                cancelAnimationFrame(animId);
                return;
              }
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
        lly: 3 * json_data["ll"][0][i] - 18, llz: 3 * json_data["ll"][1][i] + 23,
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
    audio.play();

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
  const faceColor = new THREE.Color(0xd39972);
  faceMaterial.color = faceColor;

  const whiteMaterial = new THREE.MeshStandardMaterial();
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
}

loader.load( '/static/roger_avatar.fbx', (object) => {
  setupAvatar(object);
  afterLoad(object);
  renderer.render(scene, camera);
});

//console.log(avatar);
//animate();
