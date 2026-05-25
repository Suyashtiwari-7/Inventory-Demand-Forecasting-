package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"

	"github.com/dmitryikh/leaves"
)

// PredictRequest matches the schema of inputs required by our LightGBM model.
type PredictRequest struct {
	StoreID           int     `json:"store_id"`
	SKUID             int     `json:"sku_id"`
	TotalPrice        float64 `json:"total_price"`
	BasePrice         float64 `json:"base_price"`
	IsFeaturedSKU     int     `json:"is_featured_sku"`
	IsDisplaySKU      int     `json:"is_display_sku"`
	Day               int     `json:"day"`
	Month             int     `json:"month"`
	Year              int     `json:"year"`
	WeekOfYear        int     `json:"week_of_year"`
	DayOfYear         int     `json:"day_of_year"`
	TotalPriceLag1    float64 `json:"total_price_lag_1"`
	TotalPriceLag2    float64 `json:"total_price_lag_2"`
	IsFeaturedSKULag1 int     `json:"is_featured_sku_lag_1"`
	IsDisplaySKULag1  int     `json:"is_display_sku_lag_1"`
	DiscountRatioLag1 float64 `json:"discount_ratio_lag_1"`
}

// PredictResponse contains the predicted weekly demand value.
type PredictResponse struct {
	PredictedDemand float64 `json:"predicted_demand"`
	Status          string  `json:"status"`
}

// Mappings stores the OOF target encoding averages and category lists exported from Python.
type Mappings struct {
	GlobalMean      float64            `json:"global_mean"`
	StoreCategories []int              `json:"store_categories"`
	SKUCategories   []int              `json:"sku_categories"`
	StoreSKUMap     map[string]float64 `json:"store_sku_map"`
	StoreMap        map[string]float64 `json:"store_map"`
	SKUMap          map[string]float64 `json:"sku_map"`
}

var (
	model    *leaves.Ensemble
	mappings Mappings
)

func main() {
	// 1. Load LightGBM model from native text file
	modelPath := "../models/lgbm_model.txt"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		modelPath = "models/lgbm_model.txt" // Fallback to current directory
	}

	log.Printf("Loading LightGBM model from %s...", modelPath)
	var err error
	model, err = leaves.LoadLightGBM(modelPath)
	if err != nil {
		log.Fatalf("Failed to load LightGBM model: %v", err)
	}
	log.Println("LightGBM model successfully loaded!")

	// 2. Load JSON target encodings and category mappings
	mappingsPath := "../models/mappings.json"
	if _, err := os.Stat(mappingsPath); os.IsNotExist(err) {
		mappingsPath = "models/mappings.json"
	}

	log.Printf("Loading target encodings from %s...", mappingsPath)
	mappingsFile, err := os.Open(mappingsPath)
	if err != nil {
		log.Fatalf("Failed to open mappings file: %v", err)
	}
	defer mappingsFile.Close()

	byteValue, _ := io.ReadAll(mappingsFile)
	if err := json.Unmarshal(byteValue, &mappings); err != nil {
		log.Fatalf("Failed to parse mappings JSON: %v", err)
	}
	log.Println("Category and target encodings successfully loaded!")

	// 3. Define HTTP Handlers
	http.HandleFunc("/predict", handlePredict)
	http.HandleFunc("/health", handleHealth)

	// 4. Start Server
	port := 8080
	log.Printf("Starting demand forecasting microservice on port %d...", port)
	if err := http.ListenAndServe(fmt.Sprintf(":%d", port), nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

// getCategoryIndex searches for the category in the training list and returns its index.
// Returns -1.0 if not found, which is LightGBM's standard way to handle unseen categories.
func getCategoryIndex(val int, categories []int) float64 {
	for idx, cat := range categories {
		if cat == val {
			return float64(idx)
		}
	}
	return -1.0
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status": "healthy"}`))
}

func handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error": "Only POST requests are allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, `{"error": "Failed to read request body"}`, http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var req PredictRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, `{"error": "Invalid request JSON"}`, http.StatusBadRequest)
		return
	}

	// 1. Calculate engineered promotion features
	priceDifference := req.BasePrice - req.TotalPrice
	discountRatio := priceDifference / (req.BasePrice + 1e-5)
	isDiscounted := 0.0
	if req.TotalPrice < req.BasePrice {
		isDiscounted = 1.0
	}

	// 2. Resolve target encoding values (OOF store-SKU average sales)
	globalMean := mappings.GlobalMean
	
	// Store-SKU average
	storeSKUKey := fmt.Sprintf("%d_%d", req.StoreID, req.SKUID)
	storeSKUMean, ok := mappings.StoreSKUMap[storeSKUKey]
	
	// Store average fallback
	storeMean, okStore := mappings.StoreMap[strconv.Itoa(req.StoreID)]
	if !okStore {
		storeMean = globalMean
	}
	
	// SKU average fallback
	skuMean, okSKU := mappings.SKUMap[strconv.Itoa(req.SKUID)]
	if !okSKU {
		skuMean = globalMean
	}

	// Final hierarchical fallback logic
	if !ok {
		if okSKU {
			storeSKUMean = skuMean
		} else {
			storeSKUMean = globalMean
		}
	}

	// 3. Construct LightGBM feature vector (length 22) in exact training index order
	features := make([]float64, 22)
	features[0] = getCategoryIndex(req.StoreID, mappings.StoreCategories)
	features[1] = getCategoryIndex(req.SKUID, mappings.SKUCategories)
	features[2] = req.TotalPrice
	features[3] = req.BasePrice
	features[4] = float64(req.IsFeaturedSKU)
	features[5] = float64(req.IsDisplaySKU)
	features[6] = float64(req.Day)
	features[7] = float64(req.Month)
	features[8] = float64(req.Year)
	features[9] = float64(req.WeekOfYear)
	features[10] = float64(req.DayOfYear)
	features[11] = priceDifference
	features[12] = discountRatio
	features[13] = isDiscounted
	features[14] = req.TotalPriceLag1
	features[15] = req.TotalPriceLag2
	features[16] = float64(req.IsFeaturedSKULag1)
	features[17] = float64(req.IsDisplaySKULag1)
	features[18] = req.DiscountRatioLag1
	features[19] = storeSKUMean
	features[20] = storeMean
	features[21] = skuMean

	// 4. Run model inference using leaves
	pred := model.Predict(features)
	if pred < 0 {
		pred = 0.0 // Clamp negative predictions to zero demand
	}

	// 5. Send Response
	resp := PredictResponse{
		PredictedDemand: pred,
		Status:          "success",
	}

	respBytes, err := json.Marshal(resp)
	if err != nil {
		http.Error(w, `{"error": "Failed to serialize response"}`, http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(respBytes)
}
