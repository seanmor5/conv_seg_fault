defmodule ConvSegFault.MixProject do
  use Mix.Project

  def project do
    [
      app: :conv_seg_fault,
      version: "0.1.0",
      elixir: "~> 1.11",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_env: %{
        "MIX_CURRENT_PATH" => File.cwd!(),
        "ERTS_INCLUDE_DIR" => List.to_string(:code.root_dir()) <> "/erts/include"
      }
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:elixir_make, "~> 0.6"}
    ]
  end
end
