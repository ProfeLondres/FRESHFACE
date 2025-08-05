let imageBitmap = null;
let faceLandmarks = null;

const cameraBtn = document.getElementById('camera-btn');
const uploadInput = document.getElementById('upload-input');
const previewSection = document.getElementById('preview-section');
const faceCanvas = document.getElementById('face-canvas');
const analyzeBtn = document.getElementById('analyze-btn');
const loader = document.getElementById('loader');
const resultSection = document.getElementById('result-section');
const analysisResults = document.getElementById('analysis-results');
const downloadPdfBtn = document.getElementById('download-pdf-btn');
const restartBtn = document.getElementById('restart-btn');

let videoStream = null;

// --- Cámara ---
cameraBtn.onclick = async () => {
  if (videoStream) {
    videoStream.getTracks().forEach(track => track.stop());
  }
  const video = document.createElement('video');
  video.autoplay = true;
  video.width = 400;
  video.height = 400;
  previewSection.style.display = 'block';
  faceCanvas.style.display = 'none';
  loader.style.display = 'none';
  resultSection.style.display = 'none';
  document.getElementById('input-section').style.display = 'none';
  previewSection.insertBefore(video, faceCanvas);

  try {
    videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = videoStream;
    analyzeBtn.onclick = async () => {
      faceCanvas.width = video.videoWidth;
      faceCanvas.height = video.videoHeight;
      faceCanvas.getContext('2d').drawImage(video, 0, 0);
      imageBitmap = faceCanvas.getContext('2d').getImageData(0, 0, faceCanvas.width, faceCanvas.height);
      videoStream.getTracks().forEach(track => track.stop());
      video.remove();
      faceCanvas.style.display = 'block';
      await analyzeFace();
    };
  } catch (e) {
    alert('No se pudo acceder a la cámara.');
    video.remove();
    document.getElementById('input-section').style.display = 'block';
    previewSection.style.display = 'none';
  }
};

// --- Subir imagen ---
uploadInput.onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    faceCanvas.width = img.width;
    faceCanvas.height = img.height;
    faceCanvas.getContext('2d').drawImage(img, 0, 0);
    imageBitmap = faceCanvas.getContext('2d').getImageData(0, 0, faceCanvas.width, faceCanvas.height);
    previewSection.style.display = 'block';
    faceCanvas.style.display = 'block';
    loader.style.display = 'none';
    resultSection.style.display = 'none';
    document.getElementById('input-section').style.display = 'none';
  };
  img.src = URL.createObjectURL(file);
  analyzeBtn.onclick = analyzeFace;
};

// --- Análisis facial ---
async function analyzeFace() {
  loader.style.display = 'block';
  analyzeBtn.disabled = true;
  if (!window.faceMesh) {
    loader.querySelector('span').innerText = 'Cargando modelo de análisis facial...';
    window.faceMesh = await facemesh.load();
    loader.querySelector('span').innerText = 'Analizando rostro...';
  }
  // Simula retardo de análisis (por ejemplo, 2 segundos)
  setTimeout(async () => {
    try {
      // CORRECCIÓN: Pasar directamente el canvas, no un objeto
      const predictions = await window.faceMesh.estimateFaces(faceCanvas);
      loader.style.display = 'none';
      analyzeBtn.disabled = false;

      if (!predictions.length) {
        alert('No se detectó un rostro. Intenta con otra imagen.');
        return;
      }
      faceLandmarks = predictions[0].scaledMesh;
      drawZones(faceLandmarks);
      showResults(faceLandmarks);
    } catch (error) {
      console.error('Error en el análisis:', error);
      loader.style.display = 'none';
      analyzeBtn.disabled = false;
      alert('Error al analizar el rostro. Intenta con otra imagen.');
    }
  }, 2000);
}

// --- Dibuja zonas en el canvas ---
function drawZones(landmarks) {
  const ctx = faceCanvas.getContext('2d');
  ctx.save();
  ctx.globalAlpha = 0.3;

  // Zona T (frente y nariz)
  ctx.fillStyle = '#f06292';
  drawPolygon(ctx, [
    landmarks[10], landmarks[151], landmarks[9], landmarks[197], 
    landmarks[5], landmarks[4], landmarks[19], landmarks[94],
    landmarks[168], landmarks[8], landmarks[9], landmarks[151]
  ]);

  // Mejillas
  ctx.fillStyle = '#f8bbd0';
  drawPolygon(ctx, [
    landmarks[234], landmarks[93], landmarks[132], landmarks[58]
  ]);
  drawPolygon(ctx, [
    landmarks[454], landmarks[323], landmarks[361], landmarks[288]
  ]);

  // Contorno de ojos
  ctx.fillStyle = '#ffd1dc';
  drawPolygon(ctx, [
    landmarks[33], landmarks[7], landmarks[163], landmarks[144], landmarks[145], landmarks[153], landmarks[154], landmarks[155], landmarks[133]
  ]);
  drawPolygon(ctx, [
    landmarks[263], landmarks[249], landmarks[390], landmarks[373], landmarks[374], landmarks[380], landmarks[381], landmarks[382], landmarks[362]
  ]);

  // Labios
  ctx.fillStyle = '#fce4ec';
  drawPolygon(ctx, [
    landmarks[61], landmarks[146], landmarks[91], landmarks[181], landmarks[84], landmarks[17], landmarks[314], landmarks[405], landmarks[321], landmarks[375], landmarks[291], landmarks[308]
  ]);

  // Mentón
  ctx.fillStyle = '#f8bbd0';
  drawPolygon(ctx, [
    landmarks[152], landmarks[377], landmarks[400], landmarks[378], landmarks[379], landmarks[365], landmarks[397], landmarks[172]
  ]);

  ctx.restore();
}

function drawPolygon(ctx, points) {
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i][0], points[i][1]);
  }
  ctx.closePath();
  ctx.fill();
}

// --- Análisis real de imagen por zonas ---
function analizarZonasReal(landmarks) {
  const canvas = faceCanvas;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  const zonas = {
    zonaT: getZonaTLandmarks(landmarks),
    mejillas: getMejillasLandmarks(landmarks),
    ojos: getOjosLandmarks(landmarks),
    labios: getLabiosLandmarks(landmarks),
    menton: getMentonLandmarks(landmarks)
  };

  const resultados = {};
  for (const [zona, puntos] of Object.entries(zonas)) {
    const analisis = analizarZona(imageData, puntos, canvas.width, canvas.height);
    resultados[zona] = clasificarTipoPiel(analisis);
  }

  return resultados;
}

// Landmarks por zona
function getZonaTLandmarks(landmarks) {
  return [
    landmarks[10], landmarks[151], landmarks[9], landmarks[197], 
    landmarks[5], landmarks[4], landmarks[19], landmarks[94],
    landmarks[168], landmarks[8], landmarks[9], landmarks[151]
  ];
}
function getMejillasLandmarks(landmarks) {
  return [
    landmarks[234], landmarks[93], landmarks[132], landmarks[58],
    landmarks[454], landmarks[323], landmarks[361], landmarks[288]
  ];
}
function getOjosLandmarks(landmarks) {
  return [
    landmarks[33], landmarks[7], landmarks[163], landmarks[144],
    landmarks[263], landmarks[249], landmarks[390], landmarks[373]
  ];
}
function getLabiosLandmarks(landmarks) {
  return [
    landmarks[61], landmarks[146], landmarks[91], landmarks[181],
    landmarks[84], landmarks[17], landmarks[314], landmarks[405]
  ];
}
function getMentonLandmarks(landmarks) {
  return [
    landmarks[152], landmarks[377], landmarks[400], landmarks[378],
    landmarks[379], landmarks[365], landmarks[397], landmarks[172]
  ];
}

// Analizar características de una zona específica
function analizarZona(imageData, puntos, width, height) {
  const pixels = imageData.data;
  let totalR = 0, totalG = 0, totalB = 0;
  let brillo = 0;
  let pixelCount = 0;
  let sumaVarianza = 0;

  const minX = Math.max(0, Math.min(...puntos.map(p => p[0])));
  const maxX = Math.min(width, Math.max(...puntos.map(p => p[0])));
  const minY = Math.max(0, Math.min(...puntos.map(p => p[1])));
  const maxY = Math.min(height, Math.max(...puntos.map(p => p[1])));

  for (let y = minY; y < maxY; y++) {
    for (let x = minX; x < maxX; x++) {
      if (isPointInPolygon([x, y], puntos)) {
        const index = (y * width + x) * 4;
        const r = pixels[index];
        const g = pixels[index + 1];
        const b = pixels[index + 2];

        totalR += r;
        totalG += g;
        totalB += b;
        brillo += (r + g + b) / 3;
        pixelCount++;
      }
    }
  }

  if (pixelCount === 0) return { brillo: 0, rojez: 0, textura: 0 };

  const avgR = totalR / pixelCount;
  const avgG = totalG / pixelCount;
  const avgB = totalB / pixelCount;
  const avgBrillo = brillo / pixelCount;

  // Calcular varianza para textura
  for (let y = minY; y < maxY; y++) {
    for (let x = minX; x < maxX; x++) {
      if (isPointInPolygon([x, y], puntos)) {
        const index = (y * width + x) * 4;
        const r = pixels[index];
        const g = pixels[index + 1];
        const b = pixels[index + 2];
        const pixelBrillo = (r + g + b) / 3;
        sumaVarianza += Math.pow(pixelBrillo - avgBrillo, 2);
      }
    }
  }

  const textura = Math.sqrt(sumaVarianza / pixelCount);
  const rojez = avgR - (avgG + avgB) / 2;

  return {
    brillo: avgBrillo,
    rojez: rojez,
    textura: textura,
    avgR: avgR,
    avgG: avgG,
    avgB: avgB
  };
}

// Verificar si un punto está dentro de un polígono
function isPointInPolygon(point, polygon) {
  const x = point[0], y = point[1];
  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];

    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }

  return inside;
}

// Clasificar tipo de piel basado en el análisis
function clasificarTipoPiel(analisis) {
  const { brillo, rojez, textura } = analisis;

  // Umbrales para clasificación (ajustables)
  const BRILLO_ALTO = 180;
  const BRILLO_BAJO = 120;
  const ROJEZ_ALTA = 20;
  const TEXTURA_ALTA = 25;

  if (rojez > ROJEZ_ALTA || textura > TEXTURA_ALTA) {
    return 'sensible';
  } else if (brillo > BRILLO_ALTO) {
    return 'grasa';
  } else if (brillo < BRILLO_BAJO) {
    return 'seca';
  } else if (brillo > 150) {
    return 'mixta';
  } else {
    return 'normal';
  }
}

// --- Mostrar resultados y recomendaciones ---
function showResults(landmarks) {
  previewSection.style.display = 'none';
  resultSection.style.display = 'block';

  // Análisis real de la imagen
  const zonas = analizarZonasReal(landmarks);
  const recomendaciones = obtenerRecomendacionesPorZona(zonas);
  const tipoPielGeneral = determinarTipoPielGeneral(zonas);

  analysisResults.innerHTML = `
    <div style="background: linear-gradient(135deg, #fff6f9, #fce4ec); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
      <h2 style="color: #f06292; text-align: center;">✨ Análisis Facial Completo ✨</h2>
      <p style="text-align: center; color: #4a2c2a;">Tipo de piel predominante: <strong>${tipoPielGeneral}</strong></p>
    </div>
    
    <div style="background: #fff; padding: 15px; border-radius: 10px; margin-bottom: 15px; border-left: 4px solid #f8bbd0;">
      <h3 style="color: #f06292;">🎯 Recomendaciones por zona:</h3>
      <ul style="list-style: none; padding: 0;">
        ${recomendaciones.map(r => `<li style="margin: 10px 0; padding: 8px; background: #fce4ec; border-radius: 5px;">${r}</li>`).join('')}
      </ul>
    </div>
    
    <div style="background: #fff; padding: 15px; border-radius: 10px; border-left: 4px solid #f8bbd0;">
      <h3 style="color: #f06292;">📊 Análisis detallado por zona:</h3>
      <ul style="list-style: none; padding: 0;">
        <li style="margin: 5px 0;"><strong>Zona T:</strong> <span style="color: ${getColorForSkinType(zonas.zonaT)}">${zonas.zonaT}</span></li>
        <li style="margin: 5px 0;"><strong>Mejillas:</strong> <span style="color: ${getColorForSkinType(zonas.mejillas)}">${zonas.mejillas}</span></li>
        <li style="margin: 5px 0;"><strong>Contorno de ojos:</strong> <span style="color: ${getColorForSkinType(zonas.ojos)}">${zonas.ojos}</span></li>
        <li style="margin: 5px 0;"><strong>Labios:</strong> <span style="color: ${getColorForSkinType(zonas.labios)}">${zonas.labios}</span></li>
        <li style="margin: 5px 0;"><strong>Mentón:</strong> <span style="color: ${getColorForSkinType(zonas.menton)}">${zonas.menton}</span></li>
      </ul>
    </div>
    
    <div style="background: #fff6f9; padding: 10px; border-radius: 8px; margin-top: 15px; text-align: center; font-size: 0.9em; color: #b71c1c;">
      ⚠️ Este análisis es orientativo y no reemplaza una consulta dermatológica profesional.
    </div>
  `;
}

// Recomendaciones por zona
function obtenerRecomendacionesPorZona(zonas) {
  const recomendaciones = [];

  // Zona T
  if (zonas.zonaT === 'grasa') {
    recomendaciones.push('<b>Zona T:</b> Limpieza con ácido salicílico, hidratante oil-free, protección solar ligera.');
  } else if (zonas.zonaT === 'seca') {
    recomendaciones.push('<b>Zona T:</b> Hidratación intensa, evitar exfoliantes agresivos.');
  } else if (zonas.zonaT === 'mixta') {
    recomendaciones.push('<b>Zona T:</b> Control de brillo y limpieza suave.');
  } else if (zonas.zonaT === 'sensible') {
    recomendaciones.push('<b>Zona T:</b> Productos hipoalergénicos y calmantes.');
  } else {
    recomendaciones.push('<b>Zona T:</b> Rutina equilibrada y protección solar.');
  }

  // Mejillas
  if (zonas.mejillas === 'seca') {
    recomendaciones.push('<b>Mejillas:</b> Hidratación profunda con ceramidas, evitar alcohol.');
  } else if (zonas.mejillas === 'sensible') {
    recomendaciones.push('<b>Mejillas:</b> Productos calmantes, protección solar mineral.');
  } else if (zonas.mejillas === 'grasa') {
    recomendaciones.push('<b>Mejillas:</b> Limpieza oil-free y exfoliación suave.');
  } else if (zonas.mejillas === 'mixta') {
    recomendaciones.push('<b>Mejillas:</b> Hidratación ligera y control de brillo.');
  } else {
    recomendaciones.push('<b>Mejillas:</b> Hidratación y protección solar.');
  }

  // Ojos
  recomendaciones.push('<b>Contorno de ojos:</b> Usa crema ligera con cafeína y péptidos, masaje linfático diario.');

  // Labios
  recomendaciones.push('<b>Labios:</b> Bálsamo con manteca de karité, exfoliación suave semanal, protección solar.');

  // Mentón
  if (zonas.menton === 'grasa') {
    recomendaciones.push('<b>Mentón:</b> Tratamiento con retinoides o BHA, limpieza constante.');
  } else if (zonas.menton === 'seca') {
    recomendaciones.push('<b>Mentón:</b> Hidratación y evitar productos irritantes.');
  } else if (zonas.menton === 'sensible') {
    recomendaciones.push('<b>Mentón:</b> Productos calmantes y protección solar.');
  } else if (zonas.menton === 'mixta') {
    recomendaciones.push('<b>Mentón:</b> Control de sebo y limpieza suave.');
  } else {
    recomendaciones.push('<b>Mentón:</b> Rutina equilibrada y protección solar.');
  }

  return recomendaciones;
}

// Auxiliares
function determinarTipoPielGeneral(zonas) {
  const tipos = Object.values(zonas);
  const conteo = {};
  tipos.forEach(tipo => conteo[tipo] = (conteo[tipo] || 0) + 1);
  return Object.keys(conteo).reduce((a, b) => conteo[a] > conteo[b] ? a : b);
}
function getColorForSkinType(tipo) {
  const colores = {
    'normal': '#4caf50',
    'seca': '#ff9800',
    'grasa': '#2196f3',
    'mixta': '#9c27b0',
    'sensible': '#f44336'
  };
  return colores[tipo] || '#666';
}

// --- Descargar PDF ---
downloadPdfBtn.onclick = async () => {
  const resultDiv = document.getElementById('analysis-results');
  const canvas = await html2canvas(resultDiv, { backgroundColor: "#fff6f9" });
  const imgData = canvas.toDataURL('image/png');
  const pdf = new window.jspdf.jsPDF({
    orientation: 'p',
    unit: 'pt',
    format: 'a4'
  });
  const pageWidth = pdf.internal.pageSize.getWidth();
  const imgProps = pdf.getImageProperties(imgData);
  const pdfWidth = pageWidth * 0.9;
  const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
  pdf.addImage(imgData, 'PNG', pageWidth * 0.05, 40, pdfWidth, pdfHeight);
  pdf.save('analisis_facial_freshface.pdf');
};

// --- Reiniciar ---
restartBtn.onclick = () => {
  resultSection.style.display = 'none';
  document.getElementById('input-section').style.display = 'block';
  faceCanvas.getContext('2d').clearRect(0, 0, faceCanvas.width, faceCanvas.height);
  faceCanvas.style.display = 'none';
  uploadInput.value = '';
};
