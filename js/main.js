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

// Elementos de biblioteca (pueden no existir si aún no agregaste el bloque HTML)
const newCategoryNameInput = document.getElementById('new-category-name');
const addCategoryBtn = document.getElementById('add-category-btn');
const categorySelect = document.getElementById('category-select');
const categoryFilesInput = document.getElementById('category-files');
const uploadCategoryImagesBtn = document.getElementById('upload-category-images-btn');
const rebuildCentroidsBtn = document.getElementById('rebuild-centroids-btn');
const libraryStatsDiv = document.getElementById('library-stats');
const useReferenceClassifierChk = document.getElementById('use-reference-classifier');

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
    videoStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user'
      }
    });
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
    // Optimizar resolución para mejor análisis
    const maxSize = 800;
    let { width, height } = img;

    if (width > maxSize || height > maxSize) {
      const ratio = Math.min(maxSize / width, maxSize / height);
      width = Math.round(width * ratio);
      height = Math.round(height * ratio);
    }

    faceCanvas.width = width;
    faceCanvas.height = height;
    const ctx = faceCanvas.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, width, height);
    imageBitmap = ctx.getImageData(0, 0, width, height);

    previewSection.style.display = 'block';
    faceCanvas.style.display = 'block';
    loader.style.display = 'none';
    resultSection.style.display = 'none';
    document.getElementById('input-section').style.display = 'none';
  };
  img.src = URL.createObjectURL(file);
  analyzeBtn.onclick = analyzeFace;
};

// --- Análisis facial optimizado ---
async function analyzeFace() {
  loader.style.display = 'block';
  const loaderSpan = loader.querySelector('span');
  if (loaderSpan) loaderSpan.innerText = 'Cargando modelo de análisis facial...';
  analyzeBtn.disabled = true;

  if (!window.faceMesh) {
    window.faceMesh = await facemesh.load({
      maxFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7
    });
  }
  if (loaderSpan) loaderSpan.innerText = 'Analizando rostro...';

  setTimeout(async () => {
    try {
      const predictions = await window.faceMesh.estimateFaces(faceCanvas);
      loader.style.display = 'none';
      analyzeBtn.disabled = false;

      if (!predictions.length) {
        alert('No se detectó un rostro. Asegúrate de que el rostro esté bien iluminado y centrado.');
        return;
      }

      faceLandmarks = predictions[0].scaledMesh;

      // Clasificación por referencias (si está activado y hay categorías)
      let refPrediction = null;
      const useRef = !!useReferenceClassifierChk?.checked;
      if (useRef && Object.keys(library.categories).length) {
        // Asegurar centroides actualizados
        Object.keys(library.categories).forEach(cat => {
          if (!library.categories[cat].centroid) computeCentroidForCategory(cat);
        });

        const ctx = faceCanvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, faceCanvas.width, faceCanvas.height);
        const metricsByZone = analizarZonasRealWithImageData(faceLandmarks, imageData, faceCanvas.width, faceCanvas.height);
        const vec = zonasToVector(metricsByZone);
        refPrediction = classifyByCentroids(vec);
        console.log('Clasificación por referencias ->', refPrediction);
      }

      drawZones(faceLandmarks);
      showResults(faceLandmarks, refPrediction);
    } catch (error) {
      console.error('Error en el análisis:', error);
      loader.style.display = 'none';
      analyzeBtn.disabled = false;
      alert('Error al analizar el rostro. Intenta con otra imagen.');
    }
  }, 800);
}

// --- Dibuja zonas ---
function drawZones(landmarks) {
  const ctx = faceCanvas.getContext('2d');
  ctx.save();
  ctx.globalAlpha = 0.25;

  // Zona T (frente y nariz)
  ctx.fillStyle = '#f06292';
  drawPolygon(ctx, getZonaTLandmarksOptimized(landmarks));

  // Mejillas
  ctx.fillStyle = '#f8bbd0';
  drawPolygon(ctx, getMejillaIzq(landmarks));
  drawPolygon(ctx, getMejillaDer(landmarks));

  // Ojos
  ctx.fillStyle = '#ffd1dc';
  drawPolygon(ctx, getOjoIzq(landmarks));
  drawPolygon(ctx, getOjoDer(landmarks));

  // Labios
  ctx.fillStyle = '#fce4ec';
  drawPolygon(ctx, getLabiosLandmarksOptimized(landmarks));

  // Mentón
  ctx.fillStyle = '#f8bbd0';
  drawPolygon(ctx, getMentonLandmarksOptimized(landmarks));

  ctx.restore();
}

function drawPolygon(ctx, points) {
  if (!points || points.length < 3) return;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    if (points[i] && points[i][0] !== undefined && points[i][1] !== undefined) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
  }
  ctx.closePath();
  ctx.fill();
}

// --- Landmarks por zona ---
function getZonaTLandmarksOptimized(lm) {
  return [
    lm[10], lm[151], lm[9], lm[197], lm[196], lm[3], lm[51], lm[48], lm[115], lm[131],
    lm[134], lm[102], lm[49], lm[220], lm[305], lm[281], lm[5], lm[4], lm[6], lm[168], lm[8]
  ].filter(p => p && p[0] !== undefined);
}
function getMejillaIzq(lm) {
  return [
    lm[234], lm[227], lm[137], lm[177], lm[215], lm[138], lm[135], lm[31], lm[228], lm[229],
    lm[230], lm[231], lm[232], lm[233]
  ].filter(p => p && p[0] !== undefined);
}
function getMejillaDer(lm) {
  return [
    lm[454], lm[447], lm[366], lm[401], lm[435], lm[367], lm[364], lm[394], lm[395], lm[369],
    lm[396], lm[175]
  ].filter(p => p && p[0] !== undefined);
}
function getOjoIzq(lm) {
  return [
    lm[33], lm[7], lm[163], lm[144], lm[145], lm[153], lm[154], lm[155], lm[133], lm[173],
    lm[157], lm[158], lm[159], lm[160], lm[161], lm[246]
  ].filter(p => p && p[0] !== undefined);
}
function getOjoDer(lm) {
  return [
    lm[362], lm[398], lm[384], lm[385], lm[386], lm[387], lm[388], lm[466], lm[263], lm[249],
    lm[390], lm[373], lm[374], lm[380], lm[381], lm[382]
  ].filter(p => p && p[0] !== undefined);
}
function getOjosLandmarksOptimized(lm) {
  return [...getOjoIzq(lm), ...getOjoDer(lm)];
}
function getLabiosLandmarksOptimized(lm) {
  return [
    lm[61], lm[84], lm[17], lm[314], lm[405], lm[320], lm[307], lm[375], lm[321], lm[308],
    lm[324], lm[318], lm[402], lm[317], lm[14], lm[87], lm[178], lm[88], lm[95], lm[78],
    lm[191], lm[80], lm[81], lm[82], lm[13], lm[312], lm[311], lm[310], lm[415]
  ].filter(p => p && p[0] !== undefined);
}
function getMentonLandmarksOptimized(lm) {
  return [
    lm[172], lm[136], lm[150], lm[149], lm[176], lm[148], lm[152], lm[377], lm[400], lm[378],
    lm[379], lm[365], lm[397], lm[288], lm[361], lm[323], lm[454], lm[356], lm[389], lm[251],
    lm[284], lm[332], lm[297], lm[338]
  ].filter(p => p && p[0] !== undefined);
}
function getMejillasLandmarksOptimized(lm) {
  return [...getMejillaIzq(lm), ...getMejillaDer(lm)];
}

// --- Análisis avanzado de imagen por zonas ---
function analizarZonasReal(landmarks) {
  const canvas = faceCanvas;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  const zonas = {
    zonaT: getZonaTLandmarksOptimized(landmarks),
    mejillas: getMejillasLandmarksOptimized(landmarks),
    ojos: getOjosLandmarksOptimized(landmarks),
    labios: getLabiosLandmarksOptimized(landmarks),
    menton: getMentonLandmarksOptimized(landmarks)
  };

  const resultados = {};
  for (const [zona, puntos] of Object.entries(zonas)) {
    const analisis = analizarZonaAvanzado(imageData, puntos, canvas.width, canvas.height);
    resultados[zona] = clasificarTipoPielAvanzado(analisis);
  }
  return resultados;
}

// Variante que retorna métricas crudas por zona (para vectores)
function analizarZonasRealWithImageData(landmarks, imageData, width, height) {
  const zonas = {
    zonaT: getZonaTLandmarksOptimized(landmarks),
    mejillas: getMejillasLandmarksOptimized(landmarks),
    ojos: getOjosLandmarksOptimized(landmarks),
    labios: getLabiosLandmarksOptimized(landmarks),
    menton: getMentonLandmarksOptimized(landmarks)
  };
  const resultados = {};
  for (const [zona, puntos] of Object.entries(zonas)) {
    resultados[zona] = analizarZonaAvanzado(imageData, puntos, width, height);
  }
  return resultados;
}

// Análisis de zona
function analizarZonaAvanzado(imageData, puntos, width, height) {
  if (!puntos || puntos.length === 0) return { brillo: 0, rojez: 0, textura: 0, uniformidad: 0, saturacion: 0 };

  const pixels = imageData.data;
  let totalR = 0, totalG = 0, totalB = 0;
  let brillo = 0;
  let pixelCount = 0;
  let sumaVarianza = 0;
  let sumaUniformidad = 0;

  const minX = Math.max(0, Math.floor(Math.min(...puntos.map(p => p[0]))));
  const maxX = Math.min(width, Math.ceil(Math.max(...puntos.map(p => p[0]))));
  const minY = Math.max(0, Math.floor(Math.min(...puntos.map(p => p[1]))));
  const maxY = Math.min(height, Math.ceil(Math.max(...puntos.map(p => p[1]))));

  // Promedios
  for (let y = minY; y < maxY; y++) {
    for (let x = minX; x < maxX; x++) {
      if (isPointInPolygon([x, y], puntos)) {
        const index = (y * width + x) * 4;
        const r = pixels[index];
        const g = pixels[index + 1];
        const b = pixels[index + 2];
        const luminancia = 0.299 * r + 0.587 * g + 0.114 * b;

        // Filtro para evitar extremos
        if (luminancia > 30 && luminancia < 240) {
          totalR += r; totalG += g; totalB += b;
          brillo += luminancia;
          pixelCount++;
        }
      }
    }
  }

  if (pixelCount === 0) return { brillo: 0, rojez: 0, textura: 0, uniformidad: 0, saturacion: 0 };

  const avgR = totalR / pixelCount;
  const avgG = totalG / pixelCount;
  const avgB = totalB / pixelCount;
  const avgBrillo = brillo / pixelCount;

  // Varianza (textura) y uniformidad
  for (let y = minY; y < maxY; y++) {
    for (let x = minX; x < maxX; x++) {
      if (isPointInPolygon([x, y], puntos)) {
        const index = (y * width + x) * 4;
        const r = pixels[index];
        const g = pixels[index + 1];
        const b = pixels[index + 2];
        const luminancia = 0.299 * r + 0.587 * g + 0.114 * b;

        if (luminancia > 30 && luminancia < 240) {
          sumaVarianza += Math.pow(luminancia - avgBrillo, 2);
          const diffR = Math.abs(r - avgR);
          const diffG = Math.abs(g - avgG);
          const diffB = Math.abs(b - avgB);
          sumaUniformidad += (diffR + diffG + diffB) / 3;
        }
      }
    }
  }

  const textura = Math.sqrt(sumaVarianza / pixelCount);
  const uniformidad = sumaUniformidad / pixelCount;

  // Rojez y saturación
  const rojez = (avgR - avgG) + (avgR - avgB) / 2;
  const maxRGB = Math.max(avgR, avgG, avgB);
  const minRGB = Math.min(avgR, avgG, avgB);
  const saturacion = maxRGB > 0 ? (maxRGB - minRGB) / maxRGB : 0;

  return {
    brillo: avgBrillo,
    rojez,
    textura,
    uniformidad,
    saturacion,
    avgR, avgG, avgB,
    pixelCount
  };
}

// Punto dentro de polígono
function isPointInPolygon(point, polygon) {
  if (!polygon || polygon.length < 3) return false;
  const x = point[0], y = point[1];
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    if (!polygon[i] || !polygon[j]) continue;
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];
    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

// Clasificación por umbrales balanceada (mejorada)
function clasificarTipoPielAvanzado(analisis) {
  const { brillo, rojez, textura, uniformidad } = analisis;
  let brilloNorm = Math.max(0, Math.min(255, brillo));
  let texturaNorm = Math.max(0, Math.min(50, textura));
  let rojezNorm = Math.max(-50, Math.min(50, rojez));
  let uniformidadNorm = Math.max(0, Math.min(50, uniformidad));

  // Sensible
  if ((rojezNorm > 18 && texturaNorm > 18) || (rojezNorm > 25)) return 'sensible';
  // Grasa
  if (brilloNorm > 170 && texturaNorm < 18 && uniformidadNorm < 18) return 'grasa';
  // Seca
  if (brilloNorm < 110 && texturaNorm < 15 && uniformidadNorm < 20) return 'seca';
  // Normal
  if (
    (brilloNorm >= 130 && brilloNorm <= 170 && texturaNorm < 20 && uniformidadNorm < 22) ||
    (brilloNorm >= 120 && brilloNorm <= 180 && texturaNorm < 18 && uniformidadNorm < 20 && rojezNorm < 15)
  ) return 'normal';
  // Mixta
  if (
    (brilloNorm >= 120 && brilloNorm <= 180 && (texturaNorm > 18 || uniformidadNorm > 20)) ||
    (brilloNorm >= 110 && brilloNorm <= 190 && texturaNorm > 15 && uniformidadNorm > 18)
  ) return 'mixta';

  if (brilloNorm > 180) return 'grasa';
  if (brilloNorm < 100) return 'seca';
  if (rojezNorm > 20) return 'sensible';
  if (texturaNorm > 20 || uniformidadNorm > 25) return 'mixta';
  return 'normal';
}

// --- Mostrar resultados (incluye predicción por referencias si existe) ---
function showResults(landmarks, refPrediction = null) {
  previewSection.style.display = 'none';
  resultSection.style.display = 'block';

  // Análisis real de la imagen (etiquetas por zona vía umbrales)
  const zonas = analizarZonasReal(landmarks);
  const recomendaciones = obtenerRecomendacionesPorZona(zonas);
  const tipoPielGeneral = determinarTipoPielGeneral(zonas);
  const confianza = calcularConfianzaAnalisis(zonas);

  const refHtml = (refPrediction && refPrediction.category) ? `
    <div style="background:#fff; padding:12px; border-radius:8px; margin-top:10px; border-left:4px solid #ffd1dc;">
      <h3 style="color:#f06292; margin:0 0 8px 0;">🧠 Clasificación por referencias</h3>
      <p style="margin:0;">Categoría más cercana: <b>${refPrediction.category}</b> (distancia coseno: ${refPrediction.distance.toFixed(3)})</p>
      <p style="margin:6px 0 0 0; font-size:0.9em; color:#666;">Cuantas más imágenes tenga cada categoría y más variadas sean, mejor será la precisión.</p>
    </div>
  ` : '';

  analysisResults.innerHTML = `
    <div style="background: linear-gradient(135deg, #fff6f9, #fce4ec); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
      <h2 style="color: #f06292; text-align: center; margin:0 0 8px 0;">✨ Análisis Facial Completo ✨</h2>
      <p style="text-align: center; color: #4a2c2a; margin:0 0 6px 0;">Tipo de piel predominante: <strong>${tipoPielGeneral}</strong></p>
      <p style="text-align: center; color: #666; font-size: 0.9em; margin:0;">Confianza del análisis: ${confianza}%</p>
    </div>

    <div style="background: #fff; padding: 15px; border-radius: 10px; margin-bottom: 15px; border-left: 4px solid #f8bbd0;">
      <h3 style="color: #f06292; margin:0 0 8px 0;">🎯 Recomendaciones por zona:</h3>
      <ul style="list-style: none; padding: 0; margin:0;">
        ${recomendaciones.map(r => `<li style="margin: 10px 0; padding: 8px; background: #fce4ec; border-radius: 5px;">${r}</li>`).join('')}
      </ul>
    </div>

    <div style="background: #fff; padding: 15px; border-radius: 10px; border-left: 4px solid #f8bbd0;">
      <h3 style="color: #f06292; margin:0 0 8px 0;">📊 Análisis detallado por zona:</h3>
      <ul style="list-style: none; padding: 0; margin:0;">
        <li style="margin: 5px 0;"><strong>Zona T:</strong> <span style="color: ${getColorForSkinType(zonas.zonaT)}">${zonas.zonaT}</span></li>
        <li style="margin: 5px 0;"><strong>Mejillas:</strong> <span style="color: ${getColorForSkinType(zonas.mejillas)}">${zonas.mejillas}</span></li>
        <li style="margin: 5px 0;"><strong>Contorno de ojos:</strong> <span style="color: ${getColorForSkinType(zonas.ojos)}">${zonas.ojos}</span></li>
        <li style="margin: 5px 0;"><strong>Labios:</strong> <span style="color: ${getColorForSkinType(zonas.labios)}">${zonas.labios}</span></li>
        <li style="margin: 5px 0;"><strong>Mentón:</strong> <span style="color: ${getColorForSkinType(zonas.menton)}">${zonas.menton}</span></li>
      </ul>
      ${refHtml}
    </div>

    <div style="background: #fff6f9; padding: 10px; border-radius: 8px; margin-top: 15px; text-align: center; font-size: 0.9em; color: #b71c1c;">
      ⚠️ Este análisis es orientativo y no reemplaza una consulta dermatológica profesional.
    </div>
  `;
}

// Confianza por consistencia de zonas
function calcularConfianzaAnalisis(zonas) {
  const valores = Object.values(zonas);
  const tiposUnicos = [...new Set(valores)];
  if (tiposUnicos.length === 1) return 95;
  if (tiposUnicos.length === 2) return 85;
  if (tiposUnicos.length === 3) return 75;
  return 65;
}

// Recomendaciones por zona (igual que antes, puedes ajustar)
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
  const colores = { normal: '#4caf50', seca: '#ff9800', grasa: '#2196f3', mixta: '#9c27b0', sensible: '#f44336' };
  return colores[tipo] || '#666';
}

// --- Descargar PDF ---
downloadPdfBtn.onclick = async () => {
  const resultDiv = document.getElementById('analysis-results');
  const canvas = await html2canvas(resultDiv, { backgroundColor: "#fff6f9" });
  const imgData = canvas.toDataURL('image/png');
  const pdf = new window.jspdf.jsPDF({ orientation: 'p', unit: 'pt', format: 'a4' });
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

/* =========================
   Biblioteca de ejemplos
   ========================= */
const library = {
  // categories: { 'seca': { vectors: [Float32Array], centroid: Float32Array } }
  categories: {}
};

// Inicializar listeners solo si existen los elementos (por si no agregaste el HTML aún)
if (addCategoryBtn && newCategoryNameInput && categorySelect && categoryFilesInput && uploadCategoryImagesBtn && rebuildCentroidsBtn && libraryStatsDiv) {
  addCategoryBtn.onclick = () => {
    const name = (newCategoryNameInput.value || '').trim().toLowerCase();
    if (!name) { alert('Ingresa un nombre de categoría'); return; }
    if (library.categories[name]) { alert('La categoría ya existe'); return; }
    library.categories[name] = { vectors: [], centroid: null };
    newCategoryNameInput.value = '';
    renderCategoryOptions();
    renderLibraryStats();
  };

  uploadCategoryImagesBtn.onclick = async () => {
    const cat = categorySelect.value;
    if (!cat) { alert('Selecciona una categoría'); return; }
    const files = Array.from(categoryFilesInput.files || []);
    if (files.length === 0) { alert('Selecciona imágenes'); return; }

    for (const file of files) {
      try {
        const vec = await computeFeatureVectorFromFile(file);
        if (vec) {
          library.categories[cat].vectors.push(vec);
        }
      } catch (e) {
        console.warn('Error procesando imagen de biblioteca:', e);
      }
    }
    computeCentroidForCategory(cat);
    renderLibraryStats();
    categoryFilesInput.value = '';
    alert(`Agregadas ${files.length} imágenes a la categoría "${cat}".`);
  };

  rebuildCentroidsBtn.onclick = () => {
    Object.keys(library.categories).forEach(computeCentroidForCategory);
    renderLibraryStats();
    alert('Índices/centroides recalculados.');
  };
}

function renderCategoryOptions() {
  const current = categorySelect.value;
  categorySelect.innerHTML = '<option value="">Selecciona categoría</option>';
  Object.keys(library.categories).forEach(cat => {
    const opt = document.createElement('option');
    opt.value = cat;
    opt.textContent = cat;
    categorySelect.appendChild(opt);
  });
  if (library.categories[current]) categorySelect.value = current;
}

function renderLibraryStats() {
  const parts = [];
  Object.entries(library.categories).forEach(([cat, data]) => {
    parts.push(`${cat}: ${data.vectors.length} imágenes, centroides: ${data.centroid ? 'sí' : 'no'}`);
  });
  libraryStatsDiv.textContent = parts.length ? parts.join(' | ') : 'Sin categorías cargadas.';
}

async function computeFeatureVectorFromFile(file) {
  const img = await loadImageFromFile(file);
  const { width, height, ctx } = drawToTempCanvas(img, 800);
  const imageData = ctx.getImageData(0, 0, width, height);

  if (!window.faceMesh) {
    window.faceMesh = await facemesh.load({
      maxFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7
    });
  }
  const predictions = await window.faceMesh.estimateFaces(ctx.canvas);
  if (!predictions.length) return null;
  const landmarks = predictions[0].scaledMesh;

  const zonasMetrics = analizarZonasRealWithImageData(landmarks, imageData, width, height);
  return zonasToVector(zonasMetrics);
}

function analizarZonasRealWithImageData(landmarks, imageData, width, height) {
  const zonas = {
    zonaT: getZonaTLandmarksOptimized(landmarks),
    mejillas: getMejillasLandmarksOptimized(landmarks),
    ojos: getOjosLandmarksOptimized(landmarks),
    labios: getLabiosLandmarksOptimized(landmarks),
    menton: getMentonLandmarksOptimized(landmarks)
  };
  const resultados = {};
  for (const [zona, puntos] of Object.entries(zonas)) {
    resultados[zona] = analizarZonaAvanzado(imageData, puntos, width, height);
  }
  return resultados;
}

function zonasToVector(metricsByZone) {
  const zonesOrder = ['zonaT', 'mejillas', 'ojos', 'labios', 'menton'];
  const feats = [];
  zonesOrder.forEach(z => {
    const m = metricsByZone[z] || {};
    feats.push(
      normalize(m.brillo, 0, 255),
      normalize(m.textura, 0, 50),
      normalize(m.rojez, -50, 50),
      normalize(m.uniformidad, 0, 50),
      normalize(m.saturacion, 0, 1)
    );
  });
  return new Float32Array(feats);
}

function normalize(v, min, max) {
  const clamped = Math.min(max, Math.max(min, v ?? min));
  return (clamped - min) / (max - min || 1);
}

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = reject;
    img.src = url;
  });
}

function drawToTempCanvas(img, maxSize=800) {
  let w = img.naturalWidth, h = img.naturalHeight;
  if (w > maxSize || h > maxSize) {
    const r = Math.min(maxSize / w, maxSize / h);
    w = Math.round(w * r); h = Math.round(h * r);
  }
  const can = document.createElement('canvas');
  can.width = w; can.height = h;
  const ctx = can.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, 0, 0, w, h);
  return { ctx, width: w, height: h, canvas: can };
}

function computeCentroidForCategory(cat) {
  const data = library.categories[cat];
  if (!data || data.vectors.length === 0) { if (data) data.centroid = null; return; }
  const dim = data.vectors[0].length;
  const sum = new Float32Array(dim);
  data.vectors.forEach(v => {
    for (let i=0; i<dim; i++) sum[i] += v[i];
  });
  for (let i=0; i<dim; i++) sum[i] /= data.vectors.length;
  data.centroid = sum;
}

function cosineDistance(a, b) {
  let dot=0, na=0, nb=0;
  for (let i=0; i<a.length; i++) {
    const ai=a[i], bi=b[i];
    dot += ai*bi; na += ai*ai; nb += bi*bi;
  }
  const denom = Math.sqrt(na)*Math.sqrt(nb) || 1e-6;
  return 1 - (dot/denom);
}

function classifyByCentroids(vec) {
  let bestCat = null;
  let bestDist = Infinity;
  Object.entries(library.categories).forEach(([cat, data]) => {
    if (!data.centroid) return;
    const d = cosineDistance(vec, data.centroid);
    if (d < bestDist) { bestDist = d; bestCat = cat; }
  });
  return { category: bestCat, distance: bestDist };
}
