package main

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/jackc/pgx/v5/pgxpool"
)

type Point struct {
	X, Y int
}

type BoundingBox struct {
	MinX, MinY, MaxX, MaxY int
}

var directions = []Point{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}

func isNoisyPixel(c color.Color) bool {
	r, g, b, _ := c.RGBA()
	return r > 0 || g > 0 || b > 0 ////////////////////////////////////////
}

func isInside(img image.Image, x, y int) bool {
	return x >= 0 && y >= 0 && x < img.Bounds().Dx() && y < img.Bounds().Dy()
}

func loadImage(path string) (image.Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var img image.Image
	if strings.ToLower(filepath.Ext(path)) == ".jpg" || strings.ToLower(filepath.Ext(path)) == ".jpeg" {
		img, err = jpeg.Decode(file)
	} else if strings.ToLower(filepath.Ext(path)) == ".png" {
		img, err = png.Decode(file)
	}
	return img, err
}

func dfs(img image.Image, x, y int, visited [][]bool) []Point {
	stack := []Point{{x, y}}
	group := []Point{}

	for len(stack) > 0 {
		pixel := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if visited[pixel.Y][pixel.X] {
			continue
		}
		visited[pixel.Y][pixel.X] = true
		group = append(group, pixel)

		for _, dir := range directions {
			newX, newY := pixel.X+dir.X, pixel.Y+dir.Y
			if isInside(img, newX, newY) && !visited[newY][newX] && isNoisyPixel(img.At(newX, newY)) {
				stack = append(stack, Point{newX, newY})
			}
		}
	}

	return group
}

func findGroups(img image.Image) [][]Point {
	visited := make([][]bool, img.Bounds().Dy())
	for i := range visited {
		visited[i] = make([]bool, img.Bounds().Dx())
	}

	var groups [][]Point
	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			if !visited[y][x] && isNoisyPixel(img.At(x, y)) {
				group := dfs(img, x, y, visited)
				groups = append(groups, group)
			}
		}
	}

	return groups
}

func distance(p1, p2 Point) int {
	return abs(p1.X-p2.X) + abs(p1.Y-p2.Y)
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func findDistances(groups [][]Point) [][]int {
	distances := make([][]int, len(groups))
	for i := range distances {
		distances[i] = make([]int, len(groups))
		for j := range distances[i] {
			distances[i][j] = 1 << 30 // initialized to a large number
		}
	}

	for i, group1 := range groups {
		for j, group2 := range groups {
			if i == j {
				continue
			}
			for _, p1 := range group1 {
				for _, p2 := range group2 {
					dist := distance(p1, p2)
					if dist < distances[i][j] {
						distances[i][j] = dist
					}
				}
			}
		}
	}

	return distances
}

func calculateBoundingBoxes(groups [][]Point) []BoundingBox {
	bboxes := make([]BoundingBox, len(groups))
	for idx, group := range groups {
		bbox := BoundingBox{1 << 30, 1 << 30, 0, 0}
		for _, point := range group {
			if point.X < bbox.MinX {
				bbox.MinX = point.X
			}
			if point.Y < bbox.MinY {
				bbox.MinY = point.Y
			}
			if point.X > bbox.MaxX {
				bbox.MaxX = point.X
			}
			if point.Y > bbox.MaxY {
				bbox.MaxY = point.Y
			}
		}
		bboxes[idx] = bbox
	}
	return bboxes
}

func findClosestNeighbor(index int, allBoxes []BoundingBox, distances [][]int) (int, int) {
	minDist := 1 << 30
	neighborIdx := -1

	current := allBoxes[index]
	for idx, box := range allBoxes {
		if distances[index][idx] == minDist {
			continue
		}

		// Check if the bounding box is a potential neighbor
		isLeft := box.MaxX < current.MinX
		isRight := box.MinX > current.MaxX
		isAbove := box.MaxY < current.MinY
		isBelow := box.MinY > current.MaxY

		if !(isLeft || isRight || isAbove || isBelow) {

		} else {
			if distances[index][idx] < minDist {
				minDist = distances[index][idx]
				neighborIdx = idx
			}
		}
	}
	return neighborIdx, minDist
}

func clearScreen() {
	var cmd *exec.Cmd

	// Depending on the OS, execute the appropriate command to clear the screen
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("cmd", "/c", "cls")
	case "linux", "darwin":
		cmd = exec.Command("clear")
	default:
		return
	}

	// Set the command output to the os.Stdout to display in terminal
	cmd.Stdout = os.Stdout
	cmd.Run()
}

type Pair struct {
	first  int
	second int
}

type HistEntry struct {
	Filename  string
	Method    string
	Distance  int
	Bin       int
	Frequency int
}

func PushToDb(conn *pgxpool.Pool, h HistEntry) {
	_, err := conn.Exec(context.Background(), "INSERT INTO distance_between_dots (filename, method, distance, bin, freq) VALUES ($1::text, $2::text, $3::integer, $4::integer, $5::integer);", h.Filename, h.Method, h.Distance, h.Bin, h.Frequency)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Exec failed: %v\n", err)
		os.Exit(1)
	}
}

func processFile(path string, conn *pgxpool.Pool, method string, distance int) error {
	histSum := make(map[int]int)
	if histSum == nil {
		histSum = make(map[int]int)
	}

	img, err := loadImage(path)
	if err != nil {
		fmt.Println(err)
		return err
	}
	groups := findGroups(img)
	distances := findDistances(groups)
	boundingBoxes := calculateBoundingBoxes(groups)

	visited := make([]bool, len(groups))

	for i := range groups {
		if visited[i] {
			continue
		}

		neighborIdx, minDist := findClosestNeighbor(i, boundingBoxes, distances)

		if neighborIdx != -1 { // If a neighbor exists
			histSum[minDist]++
		}

		visited[i] = true // Mark the current group as visited
	}
	for k, v := range histSum {
		h := HistEntry{
			filepath.Base(path), method, distance, k, v,
		}
		PushToDb(conn, h)
	}

	return nil
}

const THREAD_COUNT = 12

func processMethod(conn *pgxpool.Pool, method string, distance int) {
	var wg sync.WaitGroup
	sem := make(chan struct{}, THREAD_COUNT) // Semaphore pattern using channels

	// Channel to handle errors
	errChan := make(chan error, 1)

	err := filepath.Walk("./../dataset/"+method, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if !info.IsDir() {
			wg.Add(1)
			sem <- struct{}{} // Acquire semaphore
			go func(p string) {
				defer wg.Done()
				defer func() { <-sem }() // Release semaphore
				err := processFile(p, conn, method, distance)
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
	//dist := 0
	//allSum := 0

	url := "postgres://postgres:postgres@localhost:5432/postgres"
	conn, errDB := pgxpool.New(context.Background(), url)
	if errDB != nil {
		fmt.Fprintf(os.Stderr, "Unable to connect to database: %v\n", errDB)
		os.Exit(1)
	}
	//defer conn.Close(context.Background())

	//proccessMethod(conn, "noise_ev_2_256", 400)
	//processMethod(conn, "noise_ev_10_256", 2500)
	processMethod(conn, "gaussian_gs", 1)
	processMethod(conn, "perlin_gs", 1)
	processMethod(conn, "wavelet_gs", 1)

	/*for k, _ := range histSum {
		fmt.Println(dist/10*(k), "-", dist/10*(k+1), ": ", histSum[k].second, " (", histSum[k].second*100/allSum, "%)")
	}*/
}
