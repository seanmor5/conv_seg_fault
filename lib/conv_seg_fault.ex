defmodule ConvSegFault do
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:conv_seg_fault), 'libconv')
    :erlang.load_nif(path, 0)
    :os.set_signal(:sigchld, :default)
  end

  def conv, do: raise "failed to load implementation of conv/0"

  def conv_seg_fault, do: raise "failed to load implementation of conv_seg_fault/0"
end
