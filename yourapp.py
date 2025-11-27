import React, { useState, useEffect } from 'react';
import { Sprout, Leaf, TreePine, Flower2, ArrowRight, RefreshCw, ChevronDown } from 'lucide-react';

const SoilsParkApp = () => {
  const [page, setPage] = useState('start');
  const [showDetails, setShowDetails] = useState(false);
  const [formData, setFormData] = useState({ N: '', P: '', K: '', pH: '' });
  const [results, setResults] = useState(null);
  const [errors, setErrors] = useState({});

  // Animated floating elements
  const FloatingElements = () => {
    const elements = [
      { Icon: Leaf, delay: 0, duration: 15, top: '10%', left: '5%' },
      { Icon: Sprout, delay: 2, duration: 18, top: '20%', left: '85%' },
      { Icon: TreePine, delay: 4, duration: 20, top: '60%', left: '10%' },
      { Icon: Flower2, delay: 1, duration: 16, top: '70%', left: '80%' },
      { Icon: Leaf, delay: 3, duration: 17, top: '40%', left: '90%' },
      { Icon: Sprout, delay: 5, duration: 19, top: '85%', left: '15%' },
    ];

    return (
      <div className="fixed inset-0 pointer-events-none overflow-hidden" style={{ zIndex: 0 }}>
        {elements.map((elem, idx) => (
          <div
            key={idx}
            className="absolute opacity-10"
            style={{
              top: elem.top,
              left: elem.left,
              animation: `float ${elem.duration}s ease-in-out infinite`,
              animationDelay: `${elem.delay}s`
            }}
          >
            <elem.Icon size={60} className="text-green-600" />
          </div>
        ))}
        <style>{`
          @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            25% { transform: translateY(-30px) rotate(5deg); }
            50% { transform: translateY(-50px) rotate(-5deg); }
            75% { transform: translateY(-30px) rotate(3deg); }
          }
        `}</style>
      </div>
    );
  };

  // Mock prediction functions
  const predictSoilHealth = (N, P, K, pH) => {
    const nLevel = N < 200 ? 'low' : N <= 400 ? 'medium' : 'high';
    const pLevel = P < 15 ? 'low' : P <= 35 ? 'medium' : 'high';
    const kLevel = K < 110 ? 'low' : K <= 280 ? 'medium' : 'high';
    
    if (pH < 5.5 || pH > 8.5) {
      return { health: 'Low', reason: 'Extreme pH level - immediate correction needed.' };
    } else if (nLevel === 'low' || pLevel === 'low' || kLevel === 'low') {
      return { health: 'Low', reason: 'Severe nutrient deficiency detected.' };
    } else if (nLevel === 'high' && pLevel === 'high' && kLevel === 'high') {
      return { health: 'Healthy', reason: 'Your soil has good nutrient balance and suitable pH levels.' };
    } else {
      return { health: 'Moderate', reason: 'Your soil shows slight nutrient imbalance. Consider mild correction.' };
    }
  };

  const recommendFertilizer = (N, P, K, soilHealth) => {
    let primary = 'Balanced NPK';
    let confidence = 75;
    
    if (N < 200) {
      primary = 'Urea';
      confidence = 85;
    } else if (P < 15) {
      primary = 'DAP';
      confidence = 82;
    } else if (K < 110) {
      primary = 'MOP';
      confidence = 80;
    }
    
    if (soilHealth === 'Low' && !primary.toLowerCase().includes('organic')) {
      primary = primary + ' + Organic matter';
    }
    
    return { primary, confidence };
  };

  const getPhCategory = (pH) => {
    if (pH < 5.5) return { category: 'Highly acidic', text: 'Soil is highly acidic — mix agricultural lime.' };
    if (pH < 6.5) return { category: 'Slightly acidic', text: 'Soil slightly acidic — add agricultural lime.' };
    if (pH <= 7.5) return { category: 'Neutral', text: 'Soil is neutral — maintain with compost.' };
    if (pH <= 8.5) return { category: 'Slightly alkaline', text: 'Soil slightly alkaline — apply gypsum.' };
    return { category: 'Highly alkaline', text: 'Soil highly alkaline — add gypsum + compost.' };
  };

  const getNutrientWarnings = (N, P, K) => {
    const warnings = [];
    const nutrients = [
      { val: N, name: 'Nitrogen', nut: 'N', rec: 'Urea', low: 200, med: 400 },
      { val: P, name: 'Phosphorus', nut: 'P', rec: 'DAP', low: 15, med: 35 },
      { val: K, name: 'Potassium', nut: 'K', rec: 'MOP', low: 110, med: 280 }
    ];

    const highMessages = {
      N: 'Avoid extra urea; too much reduces flowering.',
      P: 'Avoid extra P; excess affects micronutrient uptake.',
      K: 'Avoid extra potash; excess reduces Mg/Ca uptake.'
    };

    nutrients.forEach(({ val, name, nut, rec, low, med }) => {
      if (val < low) {
        warnings.push(`${name} (Low): Add ${rec}`);
      } else if (val <= med) {
        warnings.push(`${name} (Medium): Balanced`);
      } else {
        warnings.push(`${name} (High): ${highMessages[nut]}`);
      }
    });

    return warnings;
  };

  const getIcarTip = (primary, soilHealth) => {
    if (soilHealth === 'Low') {
      return `Apply ${primary} with compost/FYM in 2–3 splits as per ICAR guidelines.`;
    } else if (soilHealth === 'Moderate') {
      return `Apply ${primary} in 2 splits and include compost.`;
    } else {
      return `Apply ${primary} once and maintain crop rotation.`;
    }
  };

  const handleSubmit = () => {
    const newErrors = {};
    
    const N = parseFloat(formData.N);
    const P = parseFloat(formData.P);
    const K = parseFloat(formData.K);
    const pH = parseFloat(formData.pH);

    if (isNaN(N) || N < 0 || N > 600) newErrors.N = 'Must be between 0-600';
    if (isNaN(P) || P < 0 || P > 120) newErrors.P = 'Must be between 0-120';
    if (isNaN(K) || K < 0 || K > 800) newErrors.K = 'Must be between 0-800';
    if (isNaN(pH) || pH < 3.5 || pH > 10) newErrors.pH = 'Must be between 3.5-10.0';

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    const soilHealthResult = predictSoilHealth(N, P, K, pH);
    const fertilizerResult = recommendFertilizer(N, P, K, soilHealthResult.health);
    const phResult = getPhCategory(pH);
    const warnings = getNutrientWarnings(N, P, K);
    const icarTip = getIcarTip(fertilizerResult.primary, soilHealthResult.health);

    setResults({
      inputs: { N, P, K, pH },
      soilHealth: soilHealthResult,
      fertilizer: fertilizerResult,
      ph: phResult,
      warnings,
      icarTip
    });

    setErrors({});
    setShowDetails(false);
    setPage('output');
  };

  const handleInputChange = (field, value) => {
    setFormData({ ...formData, [field]: value });
    if (errors[field]) {
      setErrors({ ...errors, [field]: undefined });
    }
  };

  // Start Page
  if (page === 'start') {
    return (
      <div className="min-h-screen flex items-center justify-center relative" 
           style={{ background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 50%, #f0fdf4 100%)' }}>
        <FloatingElements />
        
        <div className="text-center z-10 px-4">
          {/* Unique Title Design */}
          <div className="mb-12 relative">
            <div className="inline-block">
              <div className="flex items-center justify-center gap-3 mb-4">
                <Sprout size={60} className="text-green-600 animate-pulse" />
                <h1 className="text-8xl font-bold tracking-tight"
                    style={{
                      background: 'linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      textShadow: '0 4px 20px rgba(16, 185, 129, 0.3)',
                      fontFamily: 'Georgia, serif',
                      letterSpacing: '0.05em'
                    }}>
                  SOILS
                </h1>
              </div>
              <h1 className="text-8xl font-bold tracking-tight -mt-4"
                  style={{
                    background: 'linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    textShadow: '0 4px 20px rgba(16, 185, 129, 0.3)',
                    fontFamily: 'Georgia, serif',
                    letterSpacing: '0.05em'
                  }}>
                PARK
              </h1>
              <div className="flex items-center justify-center gap-2 mt-2">
                <Leaf size={24} className="text-green-500" />
                <p className="text-green-700 text-xl font-medium">AI-Powered Soil Health & Fertilizer Guidance</p>
                <Leaf size={24} className="text-green-500" />
              </div>
            </div>
          </div>

          {/* Start Button */}
          <button
            onClick={() => setPage('input')}
            className="group relative px-12 py-5 bg-gradient-to-r from-green-600 to-green-500 text-white rounded-full text-2xl font-semibold shadow-2xl hover:shadow-green-500/50 transition-all duration-300 hover:scale-105 overflow-hidden"
          >
            <span className="relative z-10 flex items-center gap-3">
              Start Journey
              <ArrowRight size={28} className="group-hover:translate-x-2 transition-transform" />
            </span>
            <div className="absolute inset-0 bg-gradient-to-r from-green-500 to-green-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          </button>

          <p className="mt-8 text-green-600 text-sm">Smart recommendations powered by machine learning</p>
        </div>
      </div>
    );
  }

  // Input Page
  if (page === 'input') {
    return (
      <div className="min-h-screen py-8 px-4 relative"
           style={{ background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 50%, #f0fdf4 100%)' }}>
        <FloatingElements />
        
        <div className="max-w-2xl mx-auto z-10 relative">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="flex items-center justify-center gap-2 mb-2">
              <Sprout size={40} className="text-green-600" />
              <h1 className="text-5xl font-bold"
                  style={{
                    background: 'linear-gradient(135deg, #059669 0%, #10b981 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    fontFamily: 'Georgia, serif'
                  }}>
                SOILS PARK
              </h1>
            </div>
            <p className="text-green-700 text-lg">Enter your soil test values for analysis</p>
          </div>

          {/* Input Form */}
          <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-green-100">
            <div className="space-y-6">
              <div>
                <label className="block text-green-800 font-semibold mb-2 text-lg">
                  Nitrogen (N) - kg/ha
                </label>
                <input
                  type="text"
                  value={formData.N}
                  onChange={(e) => handleInputChange('N', e.target.value)}
                  placeholder="e.g., 200"
                  className="w-full px-4 py-3 rounded-xl border-2 border-green-200 focus:border-green-500 focus:outline-none text-lg transition-colors"
                />
                {errors.N && <p className="text-red-500 text-sm mt-1">{errors.N}</p>}
                <p className="text-green-600 text-sm mt-1">Range: 0-600 kg/ha</p>
              </div>

              <div>
                <label className="block text-green-800 font-semibold mb-2 text-lg">
                  Phosphorus (P) - kg/ha
                </label>
                <input
                  type="text"
                  value={formData.P}
                  onChange={(e) => handleInputChange('P', e.target.value)}
                  placeholder="e.g., 30"
                  className="w-full px-4 py-3 rounded-xl border-2 border-green-200 focus:border-green-500 focus:outline-none text-lg transition-colors"
                />
                {errors.P && <p className="text-red-500 text-sm mt-1">{errors.P}</p>}
                <p className="text-green-600 text-sm mt-1">Range: 0-120 kg/ha</p>
              </div>

              <div>
                <label className="block text-green-800 font-semibold mb-2 text-lg">
                  Potassium (K) - kg/ha
                </label>
                <input
                  type="text"
                  value={formData.K}
                  onChange={(e) => handleInputChange('K', e.target.value)}
                  placeholder="e.g., 150"
                  className="w-full px-4 py-3 rounded-xl border-2 border-green-200 focus:border-green-500 focus:outline-none text-lg transition-colors"
                />
                {errors.K && <p className="text-red-500 text-sm mt-1">{errors.K}</p>}
                <p className="text-green-600 text-sm mt-1">Range: 0-800 kg/ha</p>
              </div>

              <div>
                <label className="block text-green-800 font-semibold mb-2 text-lg">
                  pH Value
                </label>
                <input
                  type="text"
                  value={formData.pH}
                  onChange={(e) => handleInputChange('pH', e.target.value)}
                  placeholder="e.g., 6.5"
                  className="w-full px-4 py-3 rounded-xl border-2 border-green-200 focus:border-green-500 focus:outline-none text-lg transition-colors"
                />
                {errors.pH && <p className="text-red-500 text-sm mt-1">{errors.pH}</p>}
                <p className="text-green-600 text-sm mt-1">Range: 3.5-10.0</p>
              </div>

              <button
                onClick={handleSubmit}
                className="w-full py-4 bg-gradient-to-r from-green-600 to-green-500 text-white rounded-xl text-xl font-semibold shadow-lg hover:shadow-xl hover:scale-[1.02] transition-all duration-300 flex items-center justify-center gap-2"
              >
                <Sprout size={24} />
                Analyze Soil
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Output Page
  if (page === 'output' && results) {
    const healthColor = {
      'Healthy': 'text-green-600',
      'Moderate': 'text-yellow-600',
      'Low': 'text-red-600'
    };

    const healthIcon = {
      'Healthy': '🟢',
      'Moderate': '🟡',
      'Low': '🔴'
    };

    return (
      <div className="min-h-screen py-8 px-4 relative"
           style={{ background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 50%, #f0fdf4 100%)' }}>
        <FloatingElements />
        
        <div className="max-w-6xl mx-auto z-10 relative">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="flex items-center justify-center gap-2 mb-2">
              <Sprout size={40} className="text-green-600" />
              <h1 className="text-5xl font-bold"
                  style={{
                    background: 'linear-gradient(135deg, #059669 0%, #10b981 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    fontFamily: 'Georgia, serif'
                  }}>
                SOILS PARK
              </h1>
            </div>
            <p className="text-green-700 text-lg">Analysis Results</p>
          </div>

          {/* Input Values */}
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-6 mb-6 border border-green-100">
            <h3 className="text-xl font-bold text-green-800 mb-4">📥 Input Values</h3>
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center">
                <p className="text-green-600 text-sm mb-1">Nitrogen (N)</p>
                <p className="text-2xl font-bold text-green-800">{results.inputs.N} kg/ha</p>
              </div>
              <div className="text-center">
                <p className="text-green-600 text-sm mb-1">Phosphorus (P)</p>
                <p className="text-2xl font-bold text-green-800">{results.inputs.P} kg/ha</p>
              </div>
              <div className="text-center">
                <p className="text-green-600 text-sm mb-1">Potassium (K)</p>
                <p className="text-2xl font-bold text-green-800">{results.inputs.K} kg/ha</p>
              </div>
              <div className="text-center">
                <p className="text-green-600 text-sm mb-1">pH</p>
                <p className="text-2xl font-bold text-green-800">{results.inputs.pH}</p>
              </div>
            </div>
          </div>

          {/* Primary Results */}
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-6 mb-6 border border-green-100">
            <h3 className="text-xl font-bold text-green-800 mb-6">🎯 Primary Results</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center p-4 bg-gradient-to-br from-green-50 to-green-100 rounded-xl">
                <p className="text-green-700 font-semibold mb-2">Soil Health</p>
                <p className={`text-4xl font-bold ${healthColor[results.soilHealth.health]} mb-2`}>
                  {healthIcon[results.soilHealth.health]} {results.soilHealth.health}
                </p>
                <p className="text-sm text-green-600 italic">{results.soilHealth.reason}</p>
              </div>

              <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl">
                <p className="text-blue-700 font-semibold mb-2">💊 Recommended Fertilizer</p>
                <p className="text-3xl font-bold text-blue-800 mb-2">{results.fertilizer.primary}</p>
                <p className="text-sm text-blue-600 italic">Confidence: {results.fertilizer.confidence}%</p>
              </div>

              <div className="text-center p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl">
                <p className="text-purple-700 font-semibold mb-2">🧪 pH Category</p>
                <p className="text-3xl font-bold text-purple-800 mb-2">{results.ph.category}</p>
                <p className="text-sm text-purple-600 italic">{results.ph.text}</p>
              </div>
            </div>
          </div>

          {/* Get Recommendations Button */}
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="w-full py-4 bg-gradient-to-r from-green-600 to-green-500 text-white rounded-xl text-xl font-semibold shadow-lg hover:shadow-xl transition-all duration-300 flex items-center justify-center gap-2 mb-6"
          >
            📋 Get Detailed Recommendations
            <ChevronDown 
              size={24} 
              className={`transform transition-transform duration-300 ${showDetails ? 'rotate-180' : ''}`} 
            />
          </button>

          {/* Detailed Results (Collapsible) */}
          {showDetails && (
            <div className="space-y-6 animate-slideDown">
              {/* ICAR Tips */}
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-green-100">
                <h3 className="text-xl font-bold text-green-800 mb-4">🌱 ICAR Action Plan</h3>
                <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded-r-xl mb-3">
                  <p className="text-blue-800">{results.icarTip}</p>
                </div>
                <div className="bg-purple-50 border-l-4 border-purple-500 p-4 rounded-r-xl">
                  <p className="text-purple-800"><strong>pH Management:</strong> {results.ph.text}</p>
                </div>
              </div>

              {/* Nutrient Warnings */}
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-green-100">
                <h3 className="text-xl font-bold text-green-800 mb-4">⚠️ Nutrient Analysis & Quick Actions</h3>
                <div className="space-y-2">
                  {results.warnings.map((warning, idx) => (
                    <div key={idx} className="flex items-start gap-2 p-3 bg-green-50 rounded-lg">
                      <span className="text-green-600 font-bold">•</span>
                      <p className="text-green-800">{warning}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Visual Charts */}
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-green-100">
                <h3 className="text-xl font-bold text-green-800 mb-4">📈 Visual Analysis</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-gradient-to-b from-green-100 to-green-50 rounded-xl">
                    <p className="font-semibold text-green-700 mb-2">Nitrogen</p>
                    <div className="text-4xl font-bold text-green-600">{results.inputs.N}</div>
                    <div className="text-sm text-green-500 mt-1">kg/ha</div>
                  </div>
                  <div className="text-center p-4 bg-gradient-to-b from-blue-100 to-blue-50 rounded-xl">
                    <p className="font-semibold text-blue-700 mb-2">Phosphorus</p>
                    <div className="text-4xl font-bold text-blue-600">{results.inputs.P}</div>
                    <div className="text-sm text-blue-500 mt-1">kg/ha</div>
                  </div>
                  <div className="text-center p-4 bg-gradient-to-b from-red-100 to-red-50 rounded-xl">
                    <p className="font-semibold text-red-700 mb-2">Potassium</p>
                    <div className="text-4xl font-bold text-red-600">{results.inputs.K}</div>
                    <div className="text-sm text-red-500 mt-1">kg/ha</div>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 border border-green-200 rounded-xl p-4 text-center">
                <p className="text-green-800">✅ Detailed recommendations generated. Use these results as guidance and cross-check with local agronomists for field-scale implementation.</p>
              </div>
            </div>
          )}

          {/* Navigation Button */}
          <div className="mt-6 text-center">
            <button
              onClick={() => {
                setPage('input');
                setFormData({ N: '', P: '', K: '', pH: '' });
                setShowDetails(false);
              }}
              className="px-8 py-3 bg-gradient-to-r from-green-600 to-green-500 text-white rounded-xl text-lg font-semibold shadow-lg hover:shadow-xl hover:scale-[1.02] transition-all duration-300 inline-flex items-center gap-2"
            >
              <RefreshCw size={20} />
              Analyze New Sample
            </button>
          </div>
        </div>

        <style>{`
          @keyframes slideDown {
            from {
              opacity: 0;
              transform: translateY(-20px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
          .animate-slideDown {
            animation: slideDown 0.5s ease-out;
          }
        `}</style>
      </div>
    );
  }

  return null;
};

export default SoilsParkApp;
