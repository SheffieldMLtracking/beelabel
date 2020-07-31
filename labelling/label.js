function convertJSONtoImageURL(data,drawcrosshairs) {
    img = data['photo']
    height = img.length
    width = img[0].length

    var canvas=document.createElement("canvas");
    var ctx=canvas.getContext("2d");

    // size the canvas to your desired image
    canvas.width=width;
    canvas.height=height;

    // get the imageData and pixel array from the canvas
    var imgData=ctx.getImageData(0,0,width,height);
    var imdata=imgData.data;

    // manipulate some pixel elements
    row = 0; //height-1;
    col = 0;
    
    var sum = 0;
    var count = 0;
    for( var i = 0; i < img.length; i+=1 ){
      for( var j = 0; j < img[i].length; j+=1 ){
        sum += img[i][j]; //don't forget to add the base
        count += 1;
      }
    }
    var avg = sum/count;
    var maxval = avg*2;
    $('span#scaletext').text(maxval.toFixed(2))
    scale = 255/maxval

    for(var i=0;i<imdata.length;i+=4){
        c = img[row][col]*scale;
        if (c>255) {c=255;}
        imdata[i]=c;
        imdata[i+1] = c;
        imdata[i+2] = c;
        imdata[i+3]=255; // make this pixel opaque
        col = col + 1;
        if (col>=width) {
          col = 0;
          row = row + 1;
        }
    }
    

    if (drawcrosshairs) {
        
        drawcrosshair(imdata,width,height)
        
    }

    // put the modified pixels back on the canvas
    ctx.putImageData(imgData,0,0);
    return "url('"+canvas.toDataURL()+"')";
}



image = 0;
x1 = 0;
x2 = 2048;
y1 = 0;
y2 = 1536;
boxsize=300;

$('button#next').click(function(){
image=image+1; 
refreshimages();})
$('button#last').click(function(){
image=image-1; 
refreshimages();})
$('button#reset').click(function(){
x1 = 0;
x2 = 2048;
y1 = 0;
y2 = 1536;
boxsize=300;
refreshimages();})
$('button#next10').click(function(){
image=image+10; 
x1 = 0;
x2 = 2048;
y1 = 0;
y2 = 1536;
boxsize=300;
refreshimages();})
$('button#last10').click(function(){
image=image-10; 
x1 = 0;
x2 = 2048;
y1 = 0;
y2 = 1536;
boxsize=300;
refreshimages();})



$(document).keypress(function(e) {
  if(e.which == 110) {$('button#next').trigger('click');}
  if(e.which == 78) {$('button#next10').trigger('click');}  
  if(e.which == 108) {$('button#last').trigger('click');}    
  if(e.which == 76) {$('button#last10').trigger('click');}
  if(e.which == 115) {$('button#save').trigger('click');}
  if(e.which == 114) {$('button#reset').trigger('click');}  
});




$('button#save').click(function(){
url = "http://127.0.0.1:5000/savepos/"+image+"/"+Math.round(x1)+"/"+Math.round(y1)+"/"+Math.round(x2)+"/"+Math.round(y2);
$.getJSON(url, function(data) {});  
image=image+1; 
x1 = 0;
x2 = 2048;
y1 = 0;
y2 = 1536;
boxsize=300;
refreshimages();})

function drawpixel(imdata,x,y,width,height) {
    pos = 4*(x+y*width)
    imdata[pos] = 255
    imdata[pos+1] = 255
    imdata[pos+2] = 0
}

function drawcrosshair(imdata,width,height) {
    x=width/2;
    y=height/2;
    for (xstep=x-10;xstep<x+10;xstep+=1) {
        drawpixel(imdata,xstep,y,width,height)
    }
    for (ystep=y-10;ystep<y+10;ystep+=1) {
        drawpixel(imdata,x,ystep,width,height)
    }

}


$('button.refreshimages').click(function(){refreshimages();});

$('#image').click(function(e){
var posX = $(this).offset().left, posY = $(this).offset().top;
centrex = (e.pageX - posX);
centrey = (e.pageY - posY);
centrex = x1+(x2-x1)*centrex/1024;
centrey = y1+(y2-y1)*centrey/768;
x1 = centrex-boxsize;
x2 = centrex+boxsize;
y1 = centrey-boxsize/1.3333333;
y2 = centrey+boxsize/1.3333333;
boxsize = boxsize / 2;
if (x1<0) {x1=0;}
if (y1<0) {y1=0;}
if (x2>2047) {x2=2047;}
if (y2>1535) {y2=1535;}
refreshimages();
});

$('input#maxval').bind('input',function() {refreshimages();});
function refreshimages(){
    $('span#imagenum').text(image)
    url = "http://127.0.0.1:5000/getimage/"+image+"/"+Math.round(x1)+"/"+Math.round(y1)+"/"+Math.round(x2)+"/"+Math.round(y2);
    $.getJSON(url, function(data) {$('#image').css("background-image",convertJSONtoImageURL(data)); });  
}

refreshimages();
