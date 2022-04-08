var app = require('express')();
var http = require('http').createServer(app);
var io = require('socket.io')(http, {
    cors: {
        origin: '*'
    }
});
var sizeof = require('object-sizeof');

app.get('/', function(req, res) {
    // res.sendFile(__dirname + '/index.html')
})

io.on('connection', function(socket) {
    console.log('A user connected')
    socket.on('imageid0', function(data) { // listen on client emit 'data'
        // var ret = Object.assign({}, data, {
        //     frame: Buffer.from(data.frame, 'base64').toString() // from buffer to base64 string
        // })
        var strImage = data.toString()

        // console.log('data : ', strImage)
        io.emit('imageid0', strImage); // emmit to socket
    })

    socket.on('imageid1', function(data) { // listen on client emit 'data'
        // var ret = Object.assign({}, data, {
        //     frame: Buffer.from(data.frame, 'base64').toString() // from buffer to base64 string
        // })
        var strImage = data.toString()

        // console.log('data : ', strImage)
        io.emit('imageid1', strImage); // emmit to socket
    })

    socket.on('imageid2', function(data) { // listen on client emit 'data'
        // var ret = Object.assign({}, data, {
        //     frame: Buffer.from(data.frame, 'base64').toString() // from buffer to base64 string
        // })
        var strImage = data.toString()

        // console.log('data : ', strImage)
        io.emit('imageid2', strImage); // emmit to socket
    })

    socket.on('imageid3', function(data) { // listen on client emit 'data'
        // var ret = Object.assign({}, data, {
        //     frame: Buffer.from(data.frame, 'base64').toString() // from buffer to base64 string
        // })
        var strImage = data.toString()

        // console.log('data : ', strImage)
        io.emit('imageid3', strImage); // emmit to socket
    })


    // socket.on('hellopython', (data) => {
    //     console.log('python')
    //     console.log(data)
    //     io.emit('hi', data)
    // })


    // setInterval(() => {
    //     // const frame = wCap.read();
    //     // const image = cv.imencode('.jpg', frame).toString('base64');
    //     socket.send('kuay')
    //     io.emit('hello', 'huaudddd')
    // }, 2000)

    socket.on('sayhi', (msg) => {
        console.log('sayhi : ', msg)
    })



    socket.on('disconnect', () => {
        console.log(" A user disconnect")
    })
})

http.listen(4333, function() {
    console.log('listening on *:4333');
})