// Global variables
let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;

// Schema configuration for Health Insurance dataset
const TARGET_FEATURE = 'Response'; // Binary classification target
const ID_FEATURE = 'id'; // Identifier to exclude from features
const NUMERICAL_FEATURES = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']; // Numerical features
const CATEGORICAL_FEATURES = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage']; // Categorical features

// Load data from uploaded CSV files
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }
    
    const statusDiv = document.getElementById('data-status');
    statusDiv.innerHTML = 'Loading data...';
    
    try {
        // Load training data
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);
        
        // Load test data
        const testText = await readFile(testFile);
        testData = parseCSV(testText);
        
        statusDiv.innerHTML = `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples`;
        
        // Enable the inspect button
        document.getElementById('inspect-btn').disabled = false;
    } catch (error) {
        statusDiv.innerHTML = `Error loading data: ${error.message}`;
        console.error(error);
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// Parse CSV text to array of objects
function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim() !== '');
    if (lines.length === 0) return [];
    
    // Parse headers first
    const headers = parseCSVLine(lines[0]);
    
    return lines.slice(1).map(line => {
        const values = parseCSVLine(line);
        const obj = {};
        headers.forEach((header, i) => {
            // Handle missing values (empty strings)
            obj[header] = i < values.length && values[i] !== '' ? values[i] : null;
            
            // Convert numerical values to numbers if possible
            if (obj[header] !== null && !isNaN(obj[header]) && obj[header] !== '') {
                obj[header] = parseFloat(obj[header]);
            }
        });
        return obj;
    });
}

function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            result.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }
    
    // Push the last field
    result.push(current.trim());
    
    return result;
}

// Inspect the loaded data
function inspectData() {
    if (!trainData || trainData.length === 0) {
        alert('Please load data first.');
        return;
    }
    
    // Show data preview
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));
    
    // Calculate and show data statistics
    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = '<h3>Data Statistics</h3>';
    
    const shapeInfo = `Dataset shape: ${trainData.length} rows x ${Object.keys(trainData[0]).length} columns`;
    const interestCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const interestRate = (interestCount / trainData.length * 100).toFixed(2);
    const targetInfo = `Interest rate: ${interestCount}/${trainData.length} (${interestRate}%)`;
    
    // Calculate missing values percentage for each feature
    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => {
        const missingCount = trainData.filter(row => row[feature] === null || row[feature] === undefined || row[feature] === '').length;
        const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
        missingInfo += `<li>${feature}: ${missingPercent}%</li>`;
    });
    missingInfo += '</ul>';
    
    statsDiv.innerHTML += `<p>${shapeInfo}</p><p>${targetInfo}</p>${missingInfo}`;
    
    // Create visualizations
    createVisualizations();
    
    // Enable the preprocess button
    document.getElementById('preprocess-btn').disabled = false;
}

// Create a preview table from data
function createPreviewTable(data) {
    const table = document.createElement('table');
    
    // Create header row
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            td.textContent = value !== null ? value : 'NULL';
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    
    return table;
}

// Create visualizations using tfjs-vis
function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';
    
    // Interest by Gender
    const interestByGender = {};
    trainData.forEach(row => {
        if (row.Gender && row.Response !== undefined) {
            if (!interestByGender[row.Gender]) {
                interestByGender[row.Gender] = { interested: 0, total: 0 };
            }
            interestByGender[row.Gender].total++;
            if (row.Response === 1) {
                interestByGender[row.Gender].interested++;
            }
        }
    });
    
    const genderData = [
        { index: 'Male', value: (interestByGender.Male?.interested / interestByGender.Male?.total) * 100 || 0 },
        { index: 'Female', value: (interestByGender.Female?.interested / interestByGender.Female?.total) * 100 || 0 }
    ];
    
    tfvis.render.barchart(
        { name: 'Interest Rate by Gender', tab: 'Charts' },
        genderData,
        { 
            xLabel: 'Gender', 
            yLabel: 'Interest Rate (%)',
            yAxisDomain: [0, 100],
            color: ['#FF6B6B', '#4ECDC4']
        }
    );
    
    // Interest by Vehicle Age
    const interestByVehicleAge = {};
    trainData.forEach(row => {
        if (row.Vehicle_Age !== undefined && row.Response !== undefined) {
            if (!interestByVehicleAge[row.Vehicle_Age]) {
                interestByVehicleAge[row.Vehicle_Age] = { interested: 0, total: 0 };
            }
            interestByVehicleAge[row.Vehicle_Age].total++;
            if (row.Response === 1) {
                interestByVehicleAge[row.Vehicle_Age].interested++;
            }
        }
    });
    
    const vehicleAgeData = Object.keys(interestByVehicleAge).map(age => ({
        index: age,
        value: (interestByVehicleAge[age].interested / interestByVehicleAge[age].total) * 100
    }));
    
    tfvis.render.barchart(
        { name: 'Interest Rate by Vehicle Age', tab: 'Charts' },
        vehicleAgeData,
        { 
            xLabel: 'Vehicle Age', 
            yLabel: 'Interest Rate (%)',
            yAxisDomain: [0, 100],
            color: ['#45B7D1', '#96CEB4', '#FEEA00']
        }
    );
    
    // Interest by Vehicle Damage
    const interestByVehicleDamage = {};
    trainData.forEach(row => {
        if (row.Vehicle_Damage !== undefined && row.Response !== undefined) {
            if (!interestByVehicleDamage[row.Vehicle_Damage]) {
                interestByVehicleDamage[row.Vehicle_Damage] = { interested: 0, total: 0 };
            }
            interestByVehicleDamage[row.Vehicle_Damage].total++;
            if (row.Response === 1) {
                interestByVehicleDamage[row.Vehicle_Damage].interested++;
            }
        }
    });
    
    const vehicleDamageData = [
        { index: 'Damaged', value: (interestByVehicleDamage.Yes?.interested / interestByVehicleDamage.Yes?.total) * 100 || 0 },
        { index: 'Not Damaged', value: (interestByVehicleDamage.No?.interested / interestByVehicleDamage.No?.total) * 100 || 0 }
    ];
    
    tfvis.render.barchart(
        { name: 'Interest Rate by Vehicle Damage History', tab: 'Charts' },
        vehicleDamageData,
        { 
            xLabel: 'Vehicle Damage', 
            yLabel: 'Interest Rate (%)',
            yAxisDomain: [0, 100],
            color: ['#FF9999', '#99FF99']
        }
    );
    
    // Age distribution by interest
    const interestedAges = trainData.filter(row => row.Response === 1 && row.Age).map(row => row.Age);
    const notInterestedAges = trainData.filter(row => row.Response === 0 && row.Age).map(row => row.Age);
    
    const ageData = {
        values: [
            { index: 'Interested', value: interestedAges },
            { index: 'Not Interested', value: notInterestedAges }
        ]
    };
    
    tfvis.render.histogram(
        { name: 'Age Distribution by Interest', tab: 'Charts' },
        ageData,
        { 
            xLabel: 'Age',
            yLabel: 'Frequency'
        }
    );
    
    chartsDiv.innerHTML += '<p>Charts are displayed in the tfjs-vis visor. Click the button in the bottom right to view.</p>';
}

// Preprocess the data
function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first.');
        return;
    }
    
    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = 'Preprocessing data...';
    
    try {
        // Calculate imputation values from training data
        const ageMedian = calculateMedian(trainData.map(row => row.Age).filter(age => age !== null));
        const annualPremiumMedian = calculateMedian(trainData.map(row => row.Annual_Premium).filter(premium => premium !== null));
        const regionCodeMedian = calculateMedian(trainData.map(row => row.Region_Code).filter(code => code !== null));
        const policyChannelMedian = calculateMedian(trainData.map(row => row.Policy_Sales_Channel).filter(channel => channel !== null));
        
        // Preprocess training data
        preprocessedTrainData = {
            features: [],
            labels: []
        };
        
        trainData.forEach(row => {
            const features = extractFeatures(row, ageMedian, annualPremiumMedian, regionCodeMedian, policyChannelMedian);
            preprocessedTrainData.features.push(features);
            preprocessedTrainData.labels.push(row[TARGET_FEATURE]);
        });
        
        // Preprocess test data
        preprocessedTestData = {
            features: [],
            customerIds: []
        };
        
        testData.forEach(row => {
            const features = extractFeatures(row, ageMedian, annualPremiumMedian, regionCodeMedian, policyChannelMedian);
            preprocessedTestData.features.push(features);
            preprocessedTestData.customerIds.push(row[ID_FEATURE]);
        });
        
        // Convert to tensors
        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels);
        
        outputDiv.innerHTML = `
            <p>Preprocessing completed!</p>
            <p>Training features shape: ${preprocessedTrainData.features.shape}</p>
            <p>Training labels shape: ${preprocessedTrainData.labels.shape}</p>
            <p>Test features shape: [${preprocessedTestData.features.length}, ${preprocessedTestData.features[0] ? preprocessedTestData.features[0].length : 0}]</p>
        `;
        
        // Enable the create model button
        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during preprocessing: ${error.message}`;
        console.error(error);
    }
}

// Extract features from a row with imputation and normalization
function extractFeatures(row, ageMedian, annualPremiumMedian, regionCodeMedian, policyChannelMedian) {
    // Impute missing values
    const age = row.Age !== null ? row.Age : ageMedian;
    const annualPremium = row.Annual_Premium !== null ? row.Annual_Premium : annualPremiumMedian;
    const regionCode = row.Region_Code !== null ? row.Region_Code : regionCodeMedian;
    const policyChannel = row.Policy_Sales_Channel !== null ? row.Policy_Sales_Channel : policyChannelMedian;
    const vintage = row.Vintage !== null ? row.Vintage : 0;
    
    // Standardize numerical features
    const standardizedAge = (age - ageMedian) / (calculateStdDev(trainData.map(r => r.Age).filter(a => a !== null)) || 1);
    const standardizedPremium = (annualPremium - annualPremiumMedian) / (calculateStdDev(trainData.map(r => r.Annual_Premium).filter(p => p !== null)) || 1);
    const standardizedRegion = (regionCode - regionCodeMedian) / (calculateStdDev(trainData.map(r => r.Region_Code).filter(c => c !== null)) || 1);
    const standardizedChannel = (policyChannel - policyChannelMedian) / (calculateStdDev(trainData.map(r => r.Policy_Sales_Channel).filter(c => c !== null)) || 1);
    const standardizedVintage = (vintage - calculateMean(trainData.map(r => r.Vintage).filter(v => v !== null))) / (calculateStdDev(trainData.map(r => r.Vintage).filter(v => v !== null)) || 1);
    
    // One-hot encode categorical features
    const genderOneHot = oneHotEncode(row.Gender, ['Male', 'Female']);
    const drivingLicenseOneHot = oneHotEncode(row.Driving_License?.toString(), ['0', '1']);
    const previouslyInsuredOneHot = oneHotEncode(row.Previously_Insured?.toString(), ['0', '1']);
    const vehicleAgeOneHot = oneHotEncode(row.Vehicle_Age, ['< 1 Year', '1-2 Year', '> 2 Years']);
    const vehicleDamageOneHot = oneHotEncode(row.Vehicle_Damage, ['Yes', 'No']);
    
    // Start with numerical features
    let features = [
        standardizedAge,
        standardizedPremium,
        standardizedRegion,
        standardizedChannel,
        standardizedVintage
    ];
    
    // Add one-hot encoded features
    features = features.concat(
        genderOneHot, 
        drivingLicenseOneHot, 
        previouslyInsuredOneHot, 
        vehicleAgeOneHot, 
        vehicleDamageOneHot
    );
    
    // Add interaction features if enabled
    if (document.getElementById('add-interaction-features').checked) {
        const agePremiumInteraction = age * annualPremium / 1000000;
        const premiumDamageInteraction = annualPremium * (row.Vehicle_Damage === 'Yes' ? 1 : 0);
        features.push(agePremiumInteraction, premiumDamageInteraction);
    }
    
    return features;
}

// Calculate median of an array
function calculateMedian(values) {
    if (values.length === 0) return 0;
    
    const sorted = [...values].sort((a, b) => a - b);
    const half = Math.floor(sorted.length / 2);
    
    if (sorted.length % 2 === 0) {
        return (sorted[half - 1] + sorted[half]) / 2;
    }
    
    return sorted[half];
}

// Calculate mean of an array
function calculateMean(values) {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
}

// Calculate standard deviation of an array
function calculateStdDev(values) {
    if (values.length === 0) return 0;
    
    const mean = calculateMean(values);
    const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
    return Math.sqrt(variance);
}

// One-hot encode a value
function oneHotEncode(value, categories) {
    const encoding = new Array(categories.length).fill(0);
    const index = categories.indexOf(value);
    if (index !== -1) {
        encoding[index] = 1;
    }
    return encoding;
}

// Create the model
function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }
    
    const inputShape = preprocessedTrainData.features.shape[1];
    const modelType = document.getElementById('model-type').value;
    
    // Create model based on selection
    model = tf.sequential();
    
    if (modelType === 'simple') {
        // Simple model for baseline
        model.add(tf.layers.dense({
            units: 8,
            activation: 'relu',
            inputShape: [inputShape]
        }));
        
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));
    } else if (modelType === 'deep') {
        // Deeper model for better performance
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu',
            inputShape: [inputShape]
        }));
        
        model.add(tf.layers.dropout({ rate: 0.3 }));
        
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dropout({ rate: 0.2 }));
        
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));
    }
    
    // Compile the model
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy', 'precision', 'recall']
    });
    
    // Display model summary
    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<h3>Model Summary</h3>';
    
    let summaryText = `<p>Model Type: ${modelType === 'simple' ? 'Simple Neural Network' : 'Deep Neural Network'}</p>`;
    summaryText += '<ul>';
    model.layers.forEach((layer, i) => {
        summaryText += `<li>Layer ${i+1}: ${layer.getClassName()} - Output Shape: ${JSON.stringify(layer.outputShape)}</li>`;
    });
    summaryText += '</ul>';
    summaryText += `<p>Total parameters: ${model.countParams()}</p>`;
    summaryDiv.innerHTML += summaryText;
    
    // Enable the train button
    document.getElementById('train-btn').disabled = false;
}

// Train the model
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create model first.');
        return;
    }
    
    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = 'Training model...';
    
    try {
        // Split training data into train and validation sets (80/20)
        const splitIndex = Math.floor(preprocessedTrainData.features.shape[0] * 0.8);
        
        const trainFeatures = preprocessedTrainData.features.slice(0, splitIndex);
        const trainLabels = preprocessedTrainData.labels.slice(0, splitIndex);
        
        const valFeatures = preprocessedTrainData.features.slice(splitIndex);
        const valLabels = preprocessedTrainData.labels.slice(splitIndex);
        
        // Store validation data for later evaluation
        validationData = valFeatures;
        validationLabels = valLabels;
        
        // Train the model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: parseInt(document.getElementById('epochs').value),
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Performance' },
                ['loss', 'acc', 'val_loss', 'val_acc'],
                { 
                    callbacks: ['onEpochEnd'],
                    onEpochEnd: (epoch, logs) => {
                        statusDiv.innerHTML = `Epoch ${epoch + 1}/${document.getElementById('epochs').value} - loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`;
                    }
                }
            )
        });
        
        statusDiv.innerHTML += '<p>Training completed!</p>';
        
        // Make predictions on validation set for evaluation
        validationPredictions = model.predict(validationData);
        
        // Enable the threshold slider and evaluation
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('threshold-slider').addEventListener('input', updateMetrics);
        
        // Enable the predict button
        document.getElementById('predict-btn').disabled = false;
        
        // Calculate initial metrics
        updateMetrics();
    } catch (error) {
        statusDiv.innerHTML = `Error during training: ${error.message}`;
        console.error(error);
    }
}

// Update metrics based on threshold
async function updateMetrics() {
    if (!validationPredictions || !validationLabels) return;
    
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    
    // Calculate confusion matrix
    const predVals = validationPredictions.arraySync();
    const trueVals = validationLabels.arraySync();
    
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    for (let i = 0; i < predVals.length; i++) {
        const prediction = predVals[i] >= threshold ? 1 : 0;
        const actual = trueVals[i];
        
        if (prediction === 1 && actual === 1) tp++;
        else if (prediction === 0 && actual === 0) tn++;
        else if (prediction === 1 && actual === 0) fp++;
        else if (prediction === 0 && actual === 1) fn++;
    }
    
    // Update confusion matrix display
    const cmDiv = document.getElementById('confusion-matrix');
    cmDiv.innerHTML = `
        <table style="border-collapse: collapse; width: 100%;">
            <tr>
                <th style="border: 1px solid #ddd; padding: 8px;"></th>
                <th style="border: 1px solid #ddd; padding: 8px;">Predicted Interested</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Predicted Not Interested</th>
            </tr>
            <tr>
                <th style="border: 1px solid #ddd; padding: 8px;">Actual Interested</th>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #d4edda;">${tp}</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f8d7da;">${fn}</td>
            </tr>
            <tr>
                <th style="border: 1px solid #ddd; padding: 8px;">Actual Not Interested</th>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #f8d7da;">${fp}</td>
                <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #d4edda;">${tn}</td>
            </tr>
        </table>
    `;
    
    // Calculate performance metrics
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
    
    // Update performance metrics display
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML = `
        <p><strong>Accuracy:</strong> ${(accuracy * 100).toFixed(2)}%</p>
        <p><strong>Precision:</strong> ${precision.toFixed(4)}</p>
        <p><strong>Recall:</strong> ${recall.toFixed(4)}</p>
        <p><strong>F1 Score:</strong> ${f1.toFixed(4)}</p>
    `;
    
    // Calculate and plot ROC curve
    await plotROC(trueVals, predVals);
}

// Plot ROC curve
async function plotROC(trueLabels, predictions) {
    // Calculate TPR and FPR for different thresholds
    const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
    const rocData = [];
    
    thresholds.forEach(threshold => {
        let tp = 0, fn = 0, fp = 0, tn = 0;
        
        for (let i = 0; i < predictions.length; i++) {
            const prediction = predictions[i] >= threshold ? 1 : 0;
            const actual = trueLabels[i];
            
            if (actual === 1) {
                if (prediction === 1) tp++;
                else fn++;
            } else {
                if (prediction === 1) fp++;
                else tn++;
            }
        }
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
        rocData.push({ threshold, fpr, tpr });
    });
    
    // Calculate AUC (approximate using trapezoidal rule)
    let auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        auc += (rocData[i].fpr - rocData[i-1].fpr) * (rocData[i].tpr + rocData[i-1].tpr) / 2;
    }
    
    // Plot ROC curve
    tfvis.render.linechart(
        { name: 'ROC Curve', tab: 'Evaluation' },
        { values: rocData.map(d => ({ x: d.fpr, y: d.tpr })) },
        { 
            xLabel: 'False Positive Rate', 
            yLabel: 'True Positive Rate',
            series: ['ROC Curve'],
            width: 400,
            height: 400
        }
    );
    
    // Add AUC to performance metrics
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML += `<p><strong>AUC:</strong> ${auc.toFixed(4)}</p>`;
}

// Predict on test data
async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train model first.');
        return;
    }
    
    const outputDiv = document.getElementById('prediction-output');
    outputDiv.innerHTML = 'Making predictions...';
    
    try {
        // Convert test features to tensor
        const testFeatures = tf.tensor2d(preprocessedTestData.features);
        
        // Make predictions
        testPredictions = model.predict(testFeatures);
        
        // Extract prediction values
        const predValues = await testPredictions.data();
        
        // Create prediction results
        const threshold = parseFloat(document.getElementById('threshold-slider').value);
        const results = preprocessedTestData.customerIds.map((id, i) => ({
            CustomerId: id,
            Interested: predValues[i] >= threshold ? 1 : 0,
            Probability: predValues[i]
        }));
        
        // Show first 10 predictions
        outputDiv.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
        outputDiv.appendChild(createPredictionTable(results.slice(0, 10)));
        
        // Calculate summary statistics
        const interestedCount = results.filter(r => r.Interested === 1).length;
        const totalCount = results.length;
        const interestRate = (interestedCount / totalCount * 100).toFixed(2);
        
        outputDiv.innerHTML += `
            <div style="margin-top: 20px; padding: 15px; background-color: #e9f7ef; border-radius: 5px;">
                <h4>Prediction Summary</h4>
                <p>Total Customers: ${totalCount}</p>
                <p>Predicted Interested: ${interestedCount} (${interestRate}%)</p>
                <p>Threshold: ${threshold.toFixed(2)}</p>
            </div>
        `;
        
        // Enable the export button
        document.getElementById('export-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during prediction: ${error.message}`;
        console.error(error);
    }
}

// Create prediction table
function createPredictionTable(data) {
    const table = document.createElement('table');
    
    // Create header row
    const headerRow = document.createElement('tr');
    ['Customer ID', 'Interested', 'Probability'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        
        // CustomerId
        const tdId = document.createElement('td');
        tdId.textContent = row.CustomerId;
        tr.appendChild(tdId);
        
        // Interested
        const tdInterested = document.createElement('td');
        tdInterested.textContent = row.Interested;
        tdInterested.style.color = row.Interested === 1 ? 'green' : 'red';
        tdInterested.style.fontWeight = 'bold';
        tr.appendChild(tdInterested);
        
        // Probability
        const tdProb = document.createElement('td');
        const prob = typeof row.Probability === 'number' ? row.Probability : parseFloat(row.Probability);
        tdProb.textContent = prob.toFixed(4);
        // Color code based on probability
        if (prob >= 0.7) {
            tdProb.style.color = 'green';
        } else if (prob >= 0.3) {
            tdProb.style.color = 'orange';
        } else {
            tdProb.style.color = 'red';
        }
        tr.appendChild(tdProb);
        
        table.appendChild(tr);
    });
    
    return table;
}

// Export results
async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        alert('Please make predictions first.');
        return;
    }
    
    const statusDiv = document.getElementById('export-status');
    statusDiv.innerHTML = 'Exporting results...';
    
    try {
        // Get predictions
        const predValues = await testPredictions.data();
        const threshold = parseFloat(document.getElementById('threshold-slider').value);
        
        // Create submission CSV (id, Response)
        let submissionCSV = 'id,Response\n';
        preprocessedTestData.customerIds.forEach((id, i) => {
            submissionCSV += `${id},${predValues[i] >= threshold ? 1 : 0}\n`;
        });
        
        // Create probabilities CSV (id, Probability)
        let probabilitiesCSV = 'id,Probability\n';
        preprocessedTestData.customerIds.forEach((id, i) => {
            probabilitiesCSV += `${id},${predValues[i].toFixed(6)}\n`;
        });
        
        // Create customer segmentation CSV
        let segmentationCSV = 'id,Probability,Segment\n';
        preprocessedTestData.customerIds.forEach((id, i) => {
            const prob = predValues[i];
            let segment = 'Low Probability';
            if (prob >= 0.7) segment = 'High Probability';
            else if (prob >= 0.4) segment = 'Medium Probability';
            segmentationCSV += `${id},${prob.toFixed(6)},${segment}\n`;
        });
        
        // Create download links
        const submissionLink = document.createElement('a');
        submissionLink.href = URL.createObjectURL(new Blob([submissionCSV], { type: 'text/csv' }));
        submissionLink.download = 'insurance_predictions.csv';
        
        const probabilitiesLink = document.createElement('a');
        probabilitiesLink.href = URL.createObjectURL(new Blob([probabilitiesCSV], { type: 'text/csv' }));
        probabilitiesLink.download = 'prediction_probabilities.csv';
        
        const segmentationLink = document.createElement('a');
        segmentationLink.href = URL.createObjectURL(new Blob([segmentationCSV], { type: 'text/csv' }));
        segmentationLink.download = 'customer_segmentation.csv';
        
        // Trigger downloads
        submissionLink.click();
        probabilitiesLink.click();
        segmentationLink.click();
        
        // Save model
        await model.save('downloads://health-insurance-model');
        
        statusDiv.innerHTML = `
            <div style="padding: 15px; background-color: #e9f7ef; border-radius: 5px;">
                <p><strong>Export completed!</strong></p>
                <p>Downloaded files:</p>
                <ul>
                    <li>insurance_predictions.csv (Binary predictions)</li>
                    <li>prediction_probabilities.csv (Prediction probabilities)</li>
                    <li>customer_segmentation.csv (Customer segmentation)</li>
                </ul>
                <p>Model saved to browser downloads</p>
            </div>
        `;
    } catch (error) {
        statusDiv.innerHTML = `Error during export: ${error.message}`;
        console.error(error);
    }
}

// Toggle visor function
function toggleVisor() {
    const button = document.getElementById('visor-toggle-btn');
    
    if (!trainData) {
        alert('Charts are not loaded yet. Please click "Inspect Data" first.');
        return;
    }
    
    const visorInstance = tfvis.visor();
    
    if (visorInstance.isOpen()) {
        visorInstance.close();
        button.innerHTML = '<span class="icon">📊</span> Show Charts';
        button.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
    } else {
        visorInstance.open();
        recreateVisualizations();
        button.innerHTML = '<span class="icon">📊</span> Hide Charts';
        button.style.background = 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)';
    }
}

// Recreate visualizations
function recreateVisualizations() {
    if (!trainData) return;
    
    // Clear existing tabs
    const visor = tfvis.visor();
    const tabs = visor.getTabs();
    tabs.forEach(tab => {
        visor.removeTab(tab);
    });
    
    createVisualizations();
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Close visor on page load
    if (tfvis.visor().isOpen()) {
        tfvis.visor().close();
    }
});