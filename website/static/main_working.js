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


const h1 = document.getElementById("audiofeed");

async function changeText() {
    const response = await fetch("/audio_feed");

    const readableStream = response.body;
    const reader = readableStream.getReader();
    while (true) {
        const { done, value } = await reader.read();
        console.log(done);
        if (done) break;
        var text = new TextDecoder("utf-8").decode(value);
        console.log(text);
        parsed_json = JSON.parse(text);
        console.log(parsed_json);
    }
}

createOffer();
changeText();