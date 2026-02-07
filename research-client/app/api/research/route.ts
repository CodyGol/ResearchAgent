import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "";
const TIMEOUT_MS = 600000; // 10 minutes (600 seconds)

export async function POST(request: NextRequest) {
  if (!BACKEND_URL) {
    return NextResponse.json(
      { error: "Backend URL not configured" },
      { status: 500 }
    );
  }

  try {
    const body = await request.json();
    const { query } = body;

    if (!query || typeof query !== "string") {
      return NextResponse.json(
        { error: "Query is required and must be a string" },
        { status: 400 }
      );
    }

    // Create AbortController for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
    }, TIMEOUT_MS);

    try {
      // Forward request to backend with timeout
      const response = await fetch(`${BACKEND_URL}/research`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Handle backend errors
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({
          error: `Backend error: ${response.status} ${response.statusText}`,
        }));

        return NextResponse.json(
          {
            error: errorData.error || `Backend returned ${response.status}`,
            status: response.status,
          },
          { status: response.status }
        );
      }

      const data = await response.json();
      return NextResponse.json(data);
    } catch (fetchError: any) {
      clearTimeout(timeoutId);

      // Handle timeout
      if (fetchError.name === "AbortError") {
        return NextResponse.json(
          {
            error: "Request timeout: Backend took longer than 10 minutes to respond",
            timeout: true,
          },
          { status: 504 }
        );
      }

      // Handle network errors
      return NextResponse.json(
        {
          error: `Network error: ${fetchError.message}`,
        },
        { status: 500 }
      );
    }
  } catch (error: any) {
    return NextResponse.json(
      {
        error: `Invalid request: ${error.message}`,
      },
      { status: 400 }
    );
  }
}
