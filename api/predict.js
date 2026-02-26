/**
 * ARTIFEX AQI — Random Forest Prediction API (Node.js)
 * GET /api/predict?ppm=v1,v2,v3,v4,v5,v6&temp=29.8&hum=51&hour=19
 *
 * Runs the RF model entirely in Node.js — no Python needed.
 * Model is loaded from /public/model.json (committed to repo).
 */

const fs   = require('fs');
const path = require('path');

// Cache model in memory across warm invocations
let MODEL_CACHE = null;

function loadModel() {
  if (MODEL_CACHE) return MODEL_CACHE;
  const modelPath = path.join(process.cwd(), 'public', 'model.json');
  if (!fs.existsSync(modelPath)) {
    throw new Error('model.json not found. Run train_model.py and commit public/model.json');
  }
  MODEL_CACHE = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
  return MODEL_CACHE;
}

// Traverse a single decision tree
function predictTree(node, features) {
  if (node.leaf) return node.value;
  return features[node.feat] <= node.thr
    ? predictTree(node.left,  features)
    : predictTree(node.right, features);
}

// Run all trees and average
function predictForest(model, features) {
  const preds = model.trees.map(tree => predictTree(tree, features));
  const mean  = preds.reduce((a, b) => a + b, 0) / preds.length;
  const std   = Math.sqrt(preds.reduce((a, b) => a + (b - mean) ** 2, 0) / preds.length);
  return { mean, std, preds };
}

function aqiLabel(ppm) {
  if (ppm < 80)  return 'Good';
  if (ppm < 150) return 'Moderate';
  if (ppm < 250) return 'Unhealthy';
  return 'Hazardous';
}

module.exports = function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('Content-Type', 'application/json');

  try {
    const { ppm, temp = '25', hum = '50', hour = '12' } = req.query;

    if (!ppm) {
      return res.status(400).json({ error: 'Missing required param: ppm' });
    }

    const ppmValues = ppm.split(',').map(Number).filter(v => !isNaN(v));
    const model     = loadModel();
    const WINDOW    = model.window || 6;

    if (ppmValues.length < WINDOW) {
      return res.status(400).json({
        error: `Need at least ${WINDOW} PPM values, got ${ppmValues.length}`
      });
    }

    // Build feature vector — must match train_model.py exactly
    const window    = ppmValues.slice(-WINDOW);
    const rollMean  = window.reduce((a, b) => a + b, 0) / window.length;
    const rollStd   = Math.sqrt(window.reduce((a, b) => a + (b - rollMean) ** 2, 0) / window.length);
    const features  = [
      ...window,
      parseFloat(temp),
      parseFloat(hum),
      parseInt(hour),
      rollMean,
      rollStd
    ];

    const { mean, std, preds } = predictForest(model, features);
    const predicted  = Math.max(0, Math.round(mean * 10) / 10);
    const confidence = Math.max(0, Math.min(100, Math.round((1 - std / Math.max(mean, 1)) * 100 * 10) / 10));

    return res.status(200).json({
      predicted_ppm: predicted,
      status:        aqiLabel(predicted),
      method:        'Random Forest Regression',
      confidence:    confidence,
      std_dev:       Math.round(std * 100) / 100,
      meta: {
        window:     WINDOW,
        mae:        model.mae,
        r2:         model.r2,
        n_samples:  model.n_samples,
        n_trees:    model.n_trees,
        trained_at: model.trained_at,
      }
    });

  } catch (e) {
    const notFound = e.message && e.message.includes('model.json');
    return res.status(notFound ? 503 : 500).json({
      error: e.message,
      hint:  notFound ? 'Run train_model.py locally and commit public/model.json' : undefined
    });
  }
};
