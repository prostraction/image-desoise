package main

import (
	"context"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
	"path/filepath"
	"sync"

	"github.com/jackc/pgx/v5/pgxpool"
)

type HistEntry struct {
	filename   string
	method     string
	bin        int
	freq_red   int
	freq_green int
	freq_blue  int
}

func pushToDB(conn *pgxpool.Pool, h HistEntry) {
	_, err := conn.Exec(context.Background(),
		"INSERT INTO rgb_hist (filename, method, bin, freq_red, freq_green, freq_blue) VALUES ($1::text, $2::text, $3::integer, $4::integer, $5::integer, $6::integer);",
		h.filename, h.method, h.bin, h.freq_red, h.freq_green, h.freq_blue)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Exec failed: %v\n", err)
		os.Exit(1)
	}
}

func processImage(conn *pgxpool.Pool, method string, imgPath string) error {
	reader, err := os.Open(imgPath)
	if err != nil {
		log.Fatal(err)
		return err
	}
	defer reader.Close()

	m, _, err := image.Decode(reader)
	if err != nil {
		log.Fatal(err)
	}
	bounds := m.Bounds()

	var histogram [256][3]int
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := m.At(x, y).RGBA()
			histogram[r>>8][0]++
			histogram[g>>8][1]++
			histogram[b>>8][2]++
		}
	}

	for i, x := range histogram {
		h := HistEntry{
			filepath.Base(imgPath),
			method,
			i,
			x[0], x[1], x[2],
		}
		pushToDB(conn, h)
	}
	return nil
}

const THREAD_COUNT = 12

func processMethod(conn *pgxpool.Pool, method string) {
	var wg sync.WaitGroup
	sem := make(chan struct{}, THREAD_COUNT) // Semaphore pattern using channels

	// Channel to handle errors
	errChan := make(chan error, 1)

	err := filepath.Walk("./../../dataset/_generated/"+method, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if !info.IsDir() {
			wg.Add(1)
			sem <- struct{}{} // Acquire semaphore
			go func(p string) {
				defer wg.Done()
				defer func() { <-sem }() // Release semaphore
				err := processImage(conn, method, path)
				if err != nil {
					select {
					case errChan <- err:
					default:
					}
				}
			}(path)
		}
		return nil
	})

	if err != nil {
		panic(err)
	}

	// Wait for all Goroutines to finish
	wg.Wait()

	// Check if any error occurred in any of the Goroutines
	select {
	case err := <-errChan:
		panic(err)
	default:
	}
}

func main() {
	url := "postgres://postgres:postgres@localhost:5432/postgres"
	conn, err := pgxpool.New(context.Background(), url)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Unable to connect to database: %v\n", err)
		os.Exit(1)
	}
	//defer conn.Close(context.Background())
	//processMethod(conn, "gaussian_ev_0.5")
	//processMethod(conn, "iso40000-crop256")

	processMethod(conn, "noise_ev_3_256")
	//processMethod(conn, "noise_ev_20_256")
}
