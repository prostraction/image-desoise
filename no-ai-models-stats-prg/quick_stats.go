package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"strings"

	_ "image/jpeg"
	_ "image/png"

	"github.com/jackc/pgx/v4/pgxpool"
	"github.com/mjibson/go-dsp/fft"
)

var db Database

func main() {

	if err := db.Init(); err != nil {
		log.Fatal(err.Error())
	}

	directory := "./demo_nl_means_denoised/"
	outputPath := "fft_results.csv"
	file, err := os.Create(outputPath)
	if err != nil {
		fmt.Printf("Error creating output file: %v\n", err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	writer.Write([]string{"Filename", "Total Power", "Average Magnitude", "Peak Frequency Magnitude", "Spectral Centroid"})

	c := 0

	err = filepath.Walk(directory, func(path string, info os.FileInfo, err error) error {
		if c == 100 {
			log.Fatal("Done")
		}

		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}

		file, err := os.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()

		img, _, err := image.Decode(file)
		if err != nil {
			return err
		}

		fftResult := applyFFT(img)
		totalPower, avgMag, peakFreqMag, spectralCentroid := analyzeFFT(fftResult)
		db.StoreFFTtoDB("ev-2-denoised-nl", file.Name(), totalPower, avgMag, peakFreqMag, spectralCentroid)
		writer.Write([]string{info.Name(), fmt.Sprintf("%f", totalPower), fmt.Sprintf("%f", avgMag), fmt.Sprintf("%f", peakFreqMag), fmt.Sprintf("%f", spectralCentroid)})

		c++

		return nil
	})

	if err != nil {
		fmt.Printf("Error walking through directory: %v\n", err)
	}
}

func processFile(path string, info os.FileInfo, err error) error {
	if err != nil {
		return err
	}
	if info.IsDir() {
		return nil
	}

	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return err
	}

	fftResult := applyFFT(img)
	saveFFTResult(fftResult, path)
	return nil
}

func applyFFT(img image.Image) [][]complex128 {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	// Convert image to grayscale and then to a 1D slice of complex numbers
	grayImg := imageToGrayArray(img)

	// First, apply FFT to each row
	fftRows := make([][]complex128, height)
	for y := 0; y < height; y++ {
		row := make([]complex128, width)
		for x := 0; x < width; x++ {
			row[x] = complex(float64(grayImg[y][x]), 0)
		}
		fftRows[y] = fft.FFT(row)
	}

	// Now, apply FFT to each column
	result := make([][]complex128, height)
	for y := 0; y < height; y++ {
		result[y] = make([]complex128, width)
	}

	for x := 0; x < width; x++ {
		col := make([]complex128, height)
		for y := 0; y < height; y++ {
			col[y] = fftRows[y][x]
		}
		fftCol := fft.FFT(col)
		for y := 0; y < height; y++ {
			result[y][x] = fftCol[y]
		}
	}

	return result
}

func imageToGrayArray(img image.Image) [][]uint8 {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	grayArray := make([][]uint8, height)
	for y := range grayArray {
		grayArray[y] = make([]uint8, width)
		for x := range grayArray[y] {
			originalColor := img.At(x, y)
			grayColor := color.GrayModel.Convert(originalColor).(color.Gray)
			grayArray[y][x] = grayColor.Y
		}
	}
	return grayArray
}

func saveFFTResult(data [][]complex128, originalPath string) {
	width := len(data[0])
	height := len(data)
	fftImage := image.NewGray(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			mag := math.Log1p(cmplx.Abs(data[y][x])) // Logarithmic scale to enhance visibility
			scaled := uint8(mag / math.Log1p(cmplx.Abs(complex(float64(255), 0))) * 255)
			fftImage.Set(x, y, color.Gray{Y: scaled})
		}
	}

	outputPath := filepath.Join(filepath.Dir(originalPath), "fft_"+filepath.Base(originalPath)+".png")
	outputFile, err := os.Create(outputPath)
	if err != nil {
		fmt.Println("Error creating FFT output file:", err)
		return
	}
	defer outputFile.Close()

	png.Encode(outputFile, fftImage)
}

func analyzeFFT(data [][]complex128) (totalPower, avgMagnitude, peakFrequencyMagnitude, spectralCentroid float64) {
	var sumMag, sumWeightedFreq float64
	var count int
	var maxMag float64 = -1

	// Iterate over each element in the FFT result
	for y, row := range data {
		for x, value := range row {
			mag := cmplx.Abs(value) // Magnitude of the FFT coefficient
			sumMag += mag
			if mag > maxMag {
				maxMag = mag // Update max magnitude and peak frequency position
			}
			freq := math.Sqrt(float64(x*x + y*y)) // Distance to the center (frequency)
			sumWeightedFreq += freq * mag
			count++
		}
	}

	totalPower = sumMag * sumMag / float64(count) // Total Power
	avgMagnitude = sumMag / float64(count)        // Average Magnitude
	peakFrequencyMagnitude = maxMag               // Peak Frequency Magnitude
	spectralCentroid = sumWeightedFreq / sumMag   // Spectral Centroid
	return
}

type Database struct {
	con *pgxpool.Pool
}

func (db *Database) noEmptyError(err error) error {
	if err == nil {
		return nil
	}
	if strings.Contains(err.Error(), "no rows in result set") {
		return nil
	}
	return err
}

func (db *Database) createTable(query string) (err error) {
	err = db.con.QueryRow(context.Background(), query).Scan()
	if db.noEmptyError(err) != nil {
		return err
	}
	return nil
}

func (db *Database) Init() (err error) {
	DB_URL := `postgres://postgres:postgres@localhost:5432/postgres`
	db.con, err = pgxpool.Connect(context.Background(), DB_URL)
	if err != nil {
		return err
	}
	err = db.createTables()
	if err != nil {
		return err
	}
	return
}

func (db *Database) createTables() (err error) {
	if err = db.createTable(`CREATE TABLE IF NOT EXISTS fft_nl (id SERIAL PRIMARY KEY, method VARCHAR(40), filename VARCHAR(200), totalPower double precision, avgMagnitude double precision, peakFrequencyMagnitude double precision, spectralCentroid double precision);`); err != nil {
		return err
	}
	return nil
}

func (db *Database) StoreFFTtoDB(method, filename string, totalPower, avgMagnitude, peakFrequencyMagnitude, spectralCentroid float64) (err error) {
	var id int
	err = db.con.QueryRow(context.Background(), "INSERT INTO fft_nl (method, filename, totalPower, avgMagnitude, peakFrequencyMagnitude, spectralCentroid) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id", method, filename, totalPower, avgMagnitude, peakFrequencyMagnitude, spectralCentroid).Scan(&id)
	return err
}
