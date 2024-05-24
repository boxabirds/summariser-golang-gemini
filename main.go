package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	genai "github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

func printResponse(resp *genai.GenerateContentResponse) {
	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				fmt.Println(part)
			}
		}
	}
	fmt.Println("---")
}

func main() {
	// see https://ai.google.dev/gemini-api/docs/models/gemini for all the models
	DEFAULT_GEMINI_MODEL := "gemini-1.5-flash-latest" // try "models/gemini-1.5-pro-latest" or "gemini-pro"

	// Define and parse the command-line flags
	inputFile := flag.String("input-file", "", "Path to the input text file")
	inputText := flag.String("input-text", "", "Input text to summarize")
	modelString := flag.String("model", DEFAULT_GEMINI_MODEL, "Model to use for the API")
	flag.Parse()

	ctx := context.Background()

	// Define the system prompt
	systemPrompt := `You are a text summarization assistant. 
	Generate a concise summary of the given input text while preserving the key information and main points. 
	Provide the summary in three bullet points, totalling 100 words or less.`

	var userMessage string
	if *inputFile != "" {
		// Read input from file
		content, err := os.ReadFile(*inputFile)
		if err != nil {
			log.Fatalf("Error reading input file: %v\n", err)
		}
		userMessage = string(content)
	} else if *inputText != "" {
		// Use input text from command-line argument
		userMessage = *inputText
	} else {
		log.Fatal("Either input-file or input-text must be provided")
	}

	client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	model := client.GenerativeModel(*modelString)
	start := time.Now()
	resp, err := model.GenerateContent(ctx, genai.Text(systemPrompt+userMessage))
	if err != nil {
		log.Fatal(err)
	}
	elapsed := time.Since(start)
	printResponse(resp)

	// Print token usage, tokens per second, and total execution time
	fmt.Printf("\nTokens generated: %d\n", resp.UsageMetadata.CandidatesTokenCount)
	fmt.Printf("Input token count: %d\n", resp.UsageMetadata.PromptTokenCount)
	fmt.Printf("Output tokens per Second: %.2f\n", float64(resp.UsageMetadata.CandidatesTokenCount)/elapsed.Seconds())
	fmt.Printf("Total Execution Time: %s\n", elapsed)
}
